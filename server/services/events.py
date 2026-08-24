"""관리자 SSE 이벤트 브로드캐스트.

- 구독: async generator subscribe() 로 클라이언트마다 Queue 하나씩
- 송출: broadcast(event_type, payload)  — 동기 컨텍스트(신고 저장 등)에서 호출 가능
         asyncio loop에 스레드세이프하게 put_nowait 스케줄링
- 하트비트: 20초마다 'ping' 이벤트 전송 (프록시/브라우저 연결 유지)
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import AsyncGenerator, Optional

log = logging.getLogger(__name__)

_loop: Optional[asyncio.AbstractEventLoop] = None
_subscribers: set[asyncio.Queue] = set()


def init(loop: asyncio.AbstractEventLoop):
    """서버 기동 시 (lifespan) 현재 이벤트 루프 등록."""
    global _loop
    _loop = loop
    log.info("events: loop registered")


def _format(event_type: str, payload) -> str:
    data = json.dumps(payload, ensure_ascii=False, default=str)
    return f"event: {event_type}\ndata: {data}\n\n"


async def subscribe() -> AsyncGenerator[str, None]:
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    _subscribers.add(q)
    log.info(f"events: subscribe (total={len(_subscribers)})")
    try:
        yield _format("hello", {"ts": datetime.now().isoformat(timespec="seconds")})
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=20.0)
                yield msg
            except asyncio.TimeoutError:
                # 하트비트
                yield _format("ping", {"ts": datetime.now().isoformat(timespec="seconds")})
    finally:
        _subscribers.discard(q)
        log.info(f"events: unsubscribe (total={len(_subscribers)})")


def broadcast(event_type: str, payload):
    """동기 컨텍스트에서도 안전하게 호출 가능."""
    if _loop is None or not _subscribers:
        return
    msg = _format(event_type, payload)

    def _fanout():
        for q in list(_subscribers):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                log.warning("events: subscriber queue full, drop event")

    _loop.call_soon_threadsafe(_fanout)

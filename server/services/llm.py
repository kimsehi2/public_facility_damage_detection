"""LLM 래퍼 — ollama 기반 챗 호출. 민원인/관리자 분리 가능."""
import logging
import re
from typing import Optional

from config import LLM_MODEL

log = logging.getLogger(__name__)

MAX_OUTPUT_TOKENS = 2048   # 보고서 완주(다. 예상 수리비까지) + thinking 폭주 방지.


def chat(messages: list[dict], model: Optional[str] = None) -> str:
    """messages = [{'role': 'system|user|assistant', 'content': '...'}, ...].
    model 미지정 시 config.LLM_MODEL (= 관리자용 e4b) 사용.

    Qwen3 계열은 thinking 모드가 기본이라 content 가 비거나 시간이 폭주함.
    `/no_think` 프롬프트 + `think=False` 파라미터로 차단.
    """
    import ollama

    target = model or LLM_MODEL
    is_qwen3 = "qwen3" in target.lower()

    # qwen3 계열엔 마지막 user 메시지 끝에 /no_think 태그 추가
    if is_qwen3 and messages:
        last = messages[-1]
        if last.get("role") == "user" and "/no_think" not in last.get("content", ""):
            messages = messages[:-1] + [
                {**last, "content": last["content"] + " /no_think"}
            ]

    kwargs = {
        "model": target,
        "messages": messages,
        "options": {"num_predict": MAX_OUTPUT_TOKENS},
    }
    if is_qwen3:
        try:
            # ollama SDK 0.5+ 지원. 구버전엔 TypeError 나면 빼고 재시도.
            res = ollama.chat(**kwargs, think=False)
        except TypeError:
            res = ollama.chat(**kwargs)
    else:
        try:
            res = ollama.chat(**kwargs)
        except Exception as e:
            log.exception("ollama.chat 실패 (model=%s)", target)
            return (
                f"⚠️ 답변 생성 실패: {e}\n"
                f"(ollama 서버가 켜져 있고 `{target}` 모델이 받아져 있는지 확인해주세요)"
            )

    try:
        content = res["message"]["content"] or ""
    except Exception:
        content = ""

    # 혹시 남은 thinking 블록 제거
    content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
    return content.strip() or "⚠️ 응답이 비어있습니다."

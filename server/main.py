"""공공기물 파손 신고 RAG 서버 — FastAPI 진입점."""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from config import BASE_DIR, UPLOADS_DIR, API_HOST, API_PORT
from routes import pages, citizen, admin
from services import admin_auth, events

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("server")

UPLOADS_DIR.mkdir(exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("서버 시작")
    import asyncio
    events.init(asyncio.get_running_loop())
    yield
    log.info("서버 종료")

app = FastAPI(title="공공기물 파손 신고 RAG", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# 관리자 API 전역 인증 미들웨어.
# 공개 경로(/api/admin/login, /api/admin/ping)만 제외하고 X-Admin-Token 검사.
_ADMIN_PUBLIC_PATHS = {"/api/admin/login", "/api/admin/ping", "/api/admin/stream"}

@app.middleware("http")
async def admin_auth_middleware(request: Request, call_next):
    path = request.url.path
    if path.startswith("/api/admin/") and path not in _ADMIN_PUBLIC_PATHS:
        # preflight(OPTIONS)는 통과
        if request.method != "OPTIONS":
            token = request.headers.get("x-admin-token", "")
            if not admin_auth.verify_token(token):
                return JSONResponse({"detail": "관리자 인증 필요"}, status_code=401)
    return await call_next(request)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

app.include_router(pages.router)
app.include_router(citizen.router, prefix="/api/citizen", tags=["citizen"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)

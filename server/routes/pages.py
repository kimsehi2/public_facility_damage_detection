"""정적 HTML 라우팅."""
from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse

from config import BASE_DIR

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
def index():
    """민원인 채팅 UI."""
    html = BASE_DIR / "static" / "index.html"
    if html.exists():
        return FileResponse(html)
    return HTMLResponse("<h1>민원인 UI 미구현</h1>")


@router.get("/login", response_class=HTMLResponse)
def login_page():
    """민원인 로그인 페이지."""
    html = BASE_DIR / "static" / "login.html"
    if html.exists():
        return FileResponse(html)
    return HTMLResponse("<h1>로그인 UI 미구현</h1>")

@router.get("/admin", response_class=HTMLResponse)
def admin():
    """관리자 대시보드."""
    html = BASE_DIR / "static" / "admin.html"
    if html.exists():
        return FileResponse(html)
    return HTMLResponse("<h1>관리자 UI 미구현</h1>")


@router.get("/admin/login", response_class=HTMLResponse)
def admin_login_page():
    """관리자 로그인 페이지."""
    html = BASE_DIR / "static" / "admin_login.html"
    if html.exists():
        return FileResponse(html)
    return HTMLResponse("<h1>관리자 로그인 UI 미구현</h1>")

@router.get("/health")
def health():
    return {"status": "ok"}

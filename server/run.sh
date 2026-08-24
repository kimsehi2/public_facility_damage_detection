#!/bin/bash

# 공공기물 파손 신고 챗봇 실행 스크립트

echo "🏛️ 공공기물 파손 신고 챗봇을 시작합니다..."

# 가상환경 확인 및 생성
if [ ! -d "venv" ]; then
    echo "📦 가상환경을 생성합니다..."
    # Python 3.13 사용 (langchain 호환성)
    python3.10 -m venv venv || python3 -m venv venv
fi

# 가상환경 활성화
echo "🔧 가상환경을 활성화합니다..."
source venv/bin/activate

# 의존성 설치
echo "📚 필요한 패키지를 설치합니다..."
pip install -r requirements.txt

# 업로드 디렉토리 생성
echo "📁 필요한 디렉토리를 생성합니다..."
mkdir -p uploads
mkdir -p static

# 서버 실행
echo "🚀 서버를 시작합니다..."
echo "📍 채팅 UI: http://localhost:8000"
echo "📱 관리자 대시보드: http://localhost:8000/admin"
echo "📊 API 문서: http://localhost:8000/docs"
echo ""
echo "종료하려면 Ctrl+C를 누르세요."

uvicorn main:app --reload --host 0.0.0.0 --port 8000
FROM python:3.10-slim

WORKDIR /app

# 시스템 패키지 설치 (ARM 환경에서 일부 패키지 빌드에 필요)
RUN apt update && apt install -y build-essential cmake

# requirements.txt 복사
COPY ./ai_server/requirements.txt ./requirements.txt

# PyTorch ARM 호환 버전 설치
RUN pip install torch==2.0.1

# 나머지 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 전체 코드 복사
COPY ./ai_server/ /app/

# 포트 설정 (Oracle에서는 PORT 환경변수 사용 안 함 → 직접 지정)
EXPOSE 8000

# FastAPI 서버 실행
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
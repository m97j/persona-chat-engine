FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces에서 0.0.0.0:7860 사용 필수
EXPOSE 7860

CMD ["python", "app.py"]

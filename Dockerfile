FROM python:3.10

WORKDIR /app
COPY ai-server/ .

RUN pip install -r requirements.txt

CMD ["python", "app.py"]
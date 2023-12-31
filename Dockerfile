FROM python:3.11-alpine
LABEL authors="Vic Ding"

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
FROM python:3.12-slim
LABEL authors="sigvardbratlie"
WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENTRYPOINT ["python", "main.py"]
CMD []
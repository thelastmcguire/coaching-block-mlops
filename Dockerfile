FROM python:3.12-slim

WORKDIR /opt/coaching-block-mlops

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python3.12", "app.py"]

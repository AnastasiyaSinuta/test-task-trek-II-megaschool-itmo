FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x start.sh

CMD ["bash", "./start.sh"]

FROM nginx
COPY nginx.conf /etc/ngin/nginx.conf
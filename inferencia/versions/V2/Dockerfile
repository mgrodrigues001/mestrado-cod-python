FROM python:3.9

WORKDIR /app

COPY ./requerements.txt /app/requerements.txt
COPY ./API_inferencia_V2.7.py /app

RUN pip install --no-cache-dir -r requerements.txt

ENTRYPOINT [ "python3", "API_inferencia_V2.7.py" ]
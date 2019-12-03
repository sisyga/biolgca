FROM python:3.8.0-slim

RUN python3 -m pip install --upgrade pip

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY lgca/ lgca/

COPY runner.py app.py

CMD ["python3", "app.py"]

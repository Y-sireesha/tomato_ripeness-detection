FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r veg/requirements.txt

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "veg.app:app"]

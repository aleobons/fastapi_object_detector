# USAGE
# docker build -t detector_placa_veiculos_api .

FROM tensorflow/tensorflow:latest

ENV PYTHONPATH "${PYTHONPATH}:/app"
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install -r requirements.txt && \
	rm requirements.txt && \
	apt-get update && \
	apt-get install ffmpeg libsm6 libxext6  -y

EXPOSE 80

COPY ./app /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
# USAGE
# docker build -t aleobons/fastapi-object-detector:v1.0 .

FROM tensorflow/tensorflow:latest

ENV PYTHONPATH "${PYTHONPATH}:/app"
ENV PYTHONUNBUFFERED=1
ENV config_model="/config_model.json"
ENV config_output="/config_output.json"
ENV config_api="/config_api.json"

COPY requirements.txt .

RUN pip install -r requirements.txt && \
	rm requirements.txt && \
	apt-get update && \
	apt-get install ffmpeg libsm6 libxext6  -y

EXPOSE 80

COPY ./app /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
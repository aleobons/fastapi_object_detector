# BUILD
# docker build -t aleobons/fastapi-object-detector:v1.0 .
# docker run --rm -p 80:80 -v $(pwd)/configs:/configs -e config_api=/configs/config_api.json -e config_model=/configs/config_model.json -e config_output=/configs/config_output.json aleobons/fastapi-object-detector:v1.0

FROM tensorflow/tensorflow:latest

ENV PYTHONPATH "${PYTHONPATH}:/app"
ENV PYTHONUNBUFFERED=1

# path dos arquivos de configuração que devem ser passados como environments variables
ENV config_model="/config_model.json"
ENV config_output="/config_output.json"
ENV config_api="/config_api.json"

COPY requirements.txt .

# ffmeg, libsm6 e libxext6 são instalados por conta do opencv
RUN pip install -r requirements.txt && \
	rm requirements.txt && \
	apt-get update && \
	apt-get install ffmpeg libsm6 libxext6  -y

EXPOSE 80

COPY ./app /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]

# fastapi_object_detector

1 - Para começar, clone o repositório:

git clone https://github.com/aleobons/fastapi_object_detector.git

2 - Instale o Docker

3 - Crie a imagem Docker da API:

docker build -t detector_placa_veiculos_api .

** rode no diretório raiz

4 - Crie a imagem do Locust:

docker build -t locustio/locust:tensorflow .

** rode no diretório tests

5 - Crie os containers:

docker compose up

6 - Teste a API no browser:

localhost:80/docs

7- Alguns códigos Python para testar a API:

# CROP
import requests
import base64

images_file = ['file.jpg', 'file1.jpg']
data = [('images_file', open(images_file[0], "rb")), ('images_file', open(images_file[1], "rb"))]

response = requests.post('http://localhost/detector_placa_veiculos/crop', files=data)

png_original = base64.b64decode(response.json()[0][0])

with open('nome_crop.png', 'wb') as f_output:
    f_output.write(png_original)

# COORDENADAS
import requests

images_file = ['file.jpg', 'file1.jpg']
data = [('images_file', open(images_file[0], "rb")), ('images_file', open(images_file[1], "rb"))]

response = requests.post('http://localhost/detector_placa_veiculos/coordenadas', files=data)

coordenadas = response.json()

# VIS_OBJECTS
import requests

data = [('images_file', open('file.jpg', "rb"))]

response = requests.post('http://localhost/detector_placa_veiculos/vis_objects', files=data)

with open('file.png', 'wb') as fd:
    for chunk in response.iter_content(chunk_size=128):
        fd.write(chunk)
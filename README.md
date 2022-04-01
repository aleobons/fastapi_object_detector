# Detector de objetos com FastAPI

O **FastAPI** é utilizado para realizar o pré-processamento das imagens e o pós-processamento dos objetos detectados conforme o output escolhido na chamada para o API.

Para inferência é utilizado o **Tensorflow Serving** e o modelo de detecção de objetos deve ter sido criado usando o **Tensorflow Object Detection API**

É possível baixar a imagem Docker através do comando `docker pull aleobons/fastapi-object-detector:v1.0` ou realizar o build conforme o Dockerfile disponível

Estão disponíveis arquivos de configuração da API, manifestos **Kubernetes**, um modelo de *detecção de placa de veículo*s e um código de teste do serviço com o **Locust**

### Exemplos de chamadas para o API:

- CROP
```
import requests
import base64

images_file = ['file.jpg', 'file1.jpg']
data = [('images_file', open(images_file[0], "rb")), ('images_file', open(images_file[1], "rb"))]

response = requests.post('http://localhost/detector_placa_veiculos/crop', files=data)

png_original = base64.b64decode(response.json()[0][0])

with open('nome_crop.png', 'wb') as f_output:
    f_output.write(png_original)
```

- COORDENADAS
```
import requests

images_file = ['file.jpg', 'file1.jpg']
data = [('images_file', open(images_file[0], "rb")), ('images_file', open(images_file[1], "rb"))]

response = requests.post('http://localhost/detector_placa_veiculos/coordenadas', files=data)

coordenadas = response.json()
```

- VIS_OBJECTS
```
import requests

data = [('images_file', open('file.jpg', "rb"))]

response = requests.post('http://localhost/detector_placa_veiculos/vis_objects', files=data)

with open('file.png', 'wb') as fd:
    for chunk in response.iter_content(chunk_size=128):
        fd.write(chunk)
```

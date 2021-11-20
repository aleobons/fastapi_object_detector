""" Arquivo de configuração

Define diversos parâmetros evitando alterar os códigos principais
"""

import os
from api.estimators.objectdetector import ObjectDetector
import utils.read_label_map as read_label_map
from api import router

# URL para o predict do modelo
MODEL_URL = 'http://detector_placa_veiculos:8501/v1/models/detector_placa_veiculos/labels/stable:predict'

# path para os arquivos diversos
FILES_PATH = os.path.sep.join(['app', 'files'])

# path para o label_map
LABEL_MAP_PATH = os.path.sep.join([FILES_PATH, 'label_map.pbtxt'])

# define as informações úteis para a API
INFO_UTEIS = {
    ObjectDetector.Infos.INFO_LABEL_MAP: read_label_map.read_label_map(LABEL_MAP_PATH),
}

# define os outputs e os seus respectivos parâmetros
OUTPUTS = {
    ObjectDetector.Output.OUTPUT_BOXES: {
            ObjectDetector.Infos.INFO_MAX_OBJECTS: 5,
            ObjectDetector.Infos.INFO_CONFIDENCE_THRESHOLD: 0.4
    },
    ObjectDetector.Output.OUTPUT_CROPS: {
            ObjectDetector.Infos.INFO_MAX_OBJECTS: 5,
            ObjectDetector.Infos.INFO_CONFIDENCE_THRESHOLD: 0.4
    },
    ObjectDetector.Output.OUTPUT_VIS_OBJECTS: {
            ObjectDetector.Infos.INFO_MAX_OBJECTS: 5,
            ObjectDetector.Infos.INFO_CONFIDENCE_THRESHOLD: 0.4,
            ObjectDetector.Infos.INFO_SHOW_CONFIDENCE: True
    }
}

# define o nome da API
NAME_API = 'Detector de placas de veículos'

# define a descrição da API
DESCRIPTION_API = 'Serviço de detecção de placas de veículos'

# define a versão da API
VERSION_API = '0.1'

# define os endpoints da API e as tags
CHAMADAS_API = {
    0: {
        'prefix': '/detect_license_plate',
        'tag': 'detect_license_plate',
        'router': router.router
    }
}

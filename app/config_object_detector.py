import os
from utils.load_model import LoadModel
from api.estimators.objectdetector import ObjectDetector
import json

# define se vai usar GPU ou não
USE_GPU = True

# path para o diretório dos modelos
MODELS_PATH = 'api/models'
MODEL = os.path.sep.join([MODELS_PATH, 'efficientdet_d0', 'v3', 'saved_model'])

# define o tipo do modelo
TYPE_MODEL = LoadModel.MODEL_TENSORFLOW

# path para arquivos diversos
FILES_PATH = 'files'

# define os paths para informações úteis para a API
INFO_UTEIS = {
    ObjectDetector.INFO_LABEL_MAP: json.loads(open('files/label_map.json').read()),
}

OUTPUTS = {
    '0': {
        'name': ObjectDetector.OUTPUT_BOXES,
        'vars': {
            ObjectDetector.INFO_MAX_OBJECTS: 1,
            ObjectDetector.INFO_CONFIDENCE_THRESHOLD: 0.4
        }
    },
    '1': {
        'name': ObjectDetector.OUTPUT_CROPS,
        'vars': {
            ObjectDetector.INFO_MAX_OBJECTS: 1,
            ObjectDetector.INFO_CONFIDENCE_THRESHOLD: 0.4
        }
    },
    '2': {
        'name': ObjectDetector.OUTPUT_VIS_OBJECTS,
        'vars': {
            ObjectDetector.INFO_MAX_OBJECTS: 1,
            ObjectDetector.INFO_CONFIDENCE_THRESHOLD: 0.4,
            ObjectDetector.INFO_SHOW_CONFIDENCE: True
        }
    }
}

# define o nome da API
NAME_API = 'Detector de placa'

# define a descrição da API
DESCRIPTION_API = 'Serviço de detecção de placas em imagens do Alerta Brasil'

# define a versão da API
VERSION_API = '0.1'

# define as chamadas da API e as tags
CHAMADAS_API = {
    '/detect_placa': 'detect_placa'
}

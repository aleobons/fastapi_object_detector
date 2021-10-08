''' Arquivo de configuração

Define diversos parâmetros evitando alterar os códigos principais
'''

import os
from utils.load_model import LoadModel
from api.estimators.objectdetector import ObjectDetector
import utils.read_label_map as read_label_map
from api import router

# define se vai usar GPU ou não
USE_GPU = True

# path para o diretório dos modelos
MODELS_PATH = os.path.sep.join(['api', 'models'])

# modelo treinado que será utilizado
MODEL = os.path.sep.join([MODELS_PATH, 'efficientdet_d1', 'v3', 'saved_model'])

# define a forma que o modelo será carregado (diretamente pelo TENSORFLOW ou pelo KERAS)
TYPE_MODEL = LoadModel.TypeModel.MODEL_TENSORFLOW

# path para os arquivos diversos
FILES_PATH = 'files'

# define as informações úteis para a API
INFO_UTEIS = {
    ObjectDetector.Infos.INFO_LABEL_MAP: read_label_map.read_label_map('files/label_map_2.pbtxt'),
}

# define os outputs e os seus respectivos parâmetros
OUTPUTS = {
    '0': {
        'name': ObjectDetector.Output.OUTPUT_BOXES,
        'vars': {
            ObjectDetector.Infos.INFO_MAX_OBJECTS: 5,
            ObjectDetector.Infos.INFO_CONFIDENCE_THRESHOLD: 0.4
        }
    },
    '1': {
        'name': ObjectDetector.Output.OUTPUT_CROPS,
        'vars': {
            ObjectDetector.Infos.INFO_MAX_OBJECTS: 1,
            ObjectDetector.Infos.INFO_CONFIDENCE_THRESHOLD: 0.4
        }
    },
    '2': {
        'name': ObjectDetector.Output.OUTPUT_VIS_OBJECTS,
        'vars': {
            ObjectDetector.Infos.INFO_MAX_OBJECTS: 5,
            ObjectDetector.Infos.INFO_CONFIDENCE_THRESHOLD: 0.4,
            ObjectDetector.Infos.INFO_SHOW_CONFIDENCE: True
        }
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
    'OBJECTS': {
        'prefix': '/detect_license_plate',
        'tag': 'detect_license_plate',
        'router': router.router
    }
}

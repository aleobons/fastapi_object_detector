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
MODEL = os.path.sep.join([MODELS_PATH, 'centernet_resnet50_v1', 'saved_model'])

# define a forma que o modelo será carregado (diretamente pelo TENSORFLOW ou pelo KERAS)
TYPE_MODEL = LoadModel.TypeModel.MODEL_TENSORFLOW

# path para os arquivos diversos
FILES_PATH = 'files'

# define as informações úteis para a API
INFO_UTEIS = {
    ObjectDetector.INFO_LABEL_MAP: read_label_map.read_label_map('files/label_map.pbtxt'),
}

# define os outputs e os seus respectivos parâmetros
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
NAME_API = 'Detector de objetos'

# define a descrição da API
DESCRIPTION_API = 'Serviço de detecção de objetos utilizando uma rede treinada no Coco dataset'

# define a versão da API
VERSION_API = '0.1'

# define as chamadas da API e as tags
CHAMADAS_API = {
    'OBJECTS': {
        'prefix': '/detect_object',
        'tag': 'detect_object',
        'router': router.router
    }
}

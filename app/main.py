'''Arquivo principal

Ao carregar a API, esse arquivo é executado carregando o modelo e algumas variáves e configurando a API
'''

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suprime o log do Tensorflow
from fastapi import FastAPI
import api.global_variables as global_vars
import config_object_detector as config
import tensorflow as tf
from utils.load_model import LoadModel

if config.USE_GPU:
    print("[INFO] Usando GPU...")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        # Habilita a alocação de memória dinâmica da GPU para evitar erro
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("[INFO] Usando CPU...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # configura para usar a CPU mesmo se a máquina tiver GPU

# carrega a API
app = FastAPI(title=config.NAME_API, description=config.DESCRIPTION_API, version=config.VERSION_API)

# define os endpoints da API
for key_call, call in config.CHAMADAS_API.items():
    app.include_router(call['router'], prefix=call['prefix'], tags=['tag'])

# carrega o modelo que será utilizado na API
global_vars.model = LoadModel(config.MODEL, config.TYPE_MODEL).carrega_modelo()

# carrega as informações úteis que serão utilizadas na API
for key, value in config.INFO_UTEIS.items():
    global_vars.info_uteis[key] = value

# carrega os parâmetros dos outputs que estarão disponíveis
global_vars.outputs = config.OUTPUTS

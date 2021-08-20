import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from fastapi import FastAPI
import api.global_variables as global_vars
import config_object_detector as config
from api import router
import tensorflow as tf
from utils.load_model import LoadModel

tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (2)

if config.USE_GPU:
    # Enable GPU dynamic memory allocation
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("[INFO] Usando GPU...")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("[INFO] Usando CPU...")

# carrega a API
app = FastAPI(title=config.NAME_API, description=config.DESCRIPTION_API, version=config.VERSION_API)

# define as chamadas da API
for prefix, tag in config.CHAMADAS_API.items():
    app.include_router(router.router, prefix=prefix, tags=[tag])

# carrega o modelo que será utilizado na API
global_vars.model = LoadModel(config.MODEL, config.TYPE_MODEL).carrega_modelo()

# carrega as informações úteis que serão utilizadas na API
for key, value in config.INFO_UTEIS.items():
    global_vars.info_uteis[key] = value

global_vars.outputs = config.OUTPUTS

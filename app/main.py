"""Arquivo principal

Ao carregar a API, esse arquivo é executado carregando o modelo e algumas variáves e configurando a API
"""

from fastapi import FastAPI
import api.global_variables as global_vars
import config_object_detector as config

# carrega a API
app = FastAPI(title=config.NAME_API, description=config.DESCRIPTION_API, version=config.VERSION_API)

# define os endpoints da API
for key_call, call in config.CHAMADAS_API.items():
    app.include_router(call['router'], prefix=call['prefix'], tags=['tag'])

# carrega a url do modelo
global_vars.model_url = config.MODEL_URL

# carrega as informações úteis que serão utilizadas na API
for key, value in config.INFO_UTEIS.items():
    global_vars.info_uteis[key] = value

# carrega os parâmetros dos outputs que estarão disponíveis
global_vars.outputs = config.OUTPUTS

"""Arquivo principal

Ao carregar a API, esse arquivo é executado carregando algumas variáves e configurando a API
"""

from fastapi import FastAPI
from api import router
import os
import json


config = json.load(open(os.environ.get("config_api")))

# carrega a API
app = FastAPI(
    title=config["NAME_API"],
    description=config["DESCRIPTION_API"],
    version=config["VERSION_API"],
)

# define os endpoints da API
for (key_call, call), router in zip(config["CHAMADAS_API"].items(), [router.router]):
    app.include_router(router, prefix=call["prefix"], tags=["tag"])

"""Funções do router

Funções utilizadas para mapear as operações do endepoint principal da API

Uma operação para cada output do detector de objetos
"""

# importa os pacotes necessários
from fastapi import APIRouter, UploadFile, File
import io
from starlette.responses import StreamingResponse
from typing import List, Dict
import time
from fastapi import HTTPException

# importa os módulos próprios necessários
from .preprocessors.uploadimagepreprocessor import UploadImagePreprocessor
from .estimators.objectdetector import ObjectDetector as Estimator
import api.global_variables as variaveis

# módulo que lida com as diversas operações do endpoint
router = APIRouter()


@router.post("/coordenadas", status_code=200)
async def post(images_file: List[UploadFile] = File(...)) -> List[Dict]:
    """Detecta objetos e retorna as coordenadas

    Args:
        images_file: lista arquivos de imagens para o detector procurar objetos

    Returns:
        Uma lista com as coordenadas, as confianças e os rótulos dos objetos detectados em cada imagem

    """

    # define o output
    output = Estimator.Output.OUTPUT_BOXES

    # coleta as informações do output
    output_coordenadas = variaveis.outputs.get(output, None)

    # executa a predição nas imagens informando o output de coordenadas
    coordenadas = await execute(images_file, output=output, vars_output=output_coordenadas)

    return coordenadas


@router.post("/crop", status_code=200)
async def post(images_file: List[UploadFile] = File(...)):
    """Detecta objetos e retorna a recorte do objeto

        Args:
            images_file: lista arquivos de imagens para o detector procurar objetos

        Returns:
            Uma lista de objetos recortados das imagens codificadas em formato binário

        Raises:
            HTTPException: Um erro ocorre se não for detectado objeto.
        """

    # define o output
    output = Estimator.Output.OUTPUT_CROPS

    # coleta as informações do output
    output_crop = variaveis.outputs.get(output, None)

    # executa a predição nas imagens informando o output crop
    images = await execute(images_file, output=output, vars_output=output_crop)

    # lida com a não detecção de nenhum objeto com um erro
    if images is None:
        raise HTTPException(status_code=406, detail="Nenhum objeto detectado na imagem.")

    return images


@router.post("/vis_objects", status_code=200)
async def post(images_file: UploadFile = File(...)):
    """Detecta objetos e retorna a imagem com as detecções

    Args:
        images_file: arquivo de imagem para o detector procurar objetos

    Returns:
        A imagem com os objetos anotados, os rótulos e as confianças

    """

    # define o output
    output = Estimator.Output.OUTPUT_VIS_OBJECTS

    # coleta as informações do output
    output_vis_objects = variaveis.outputs.get(output, None)

    # executa a predição na imagem informando o output de visualização dos objetos. A imagem é colocada dentro de uma
    # lista pois a função execute espera uma lista
    image = await execute([images_file], output=output, vars_output=output_vis_objects)

    # O StreamingResponse é utilizado para retornar a imagem já codificada em jpg
    return StreamingResponse(io.BytesIO(image), media_type="image/jpg")


async def execute(images_file, output, vars_output):
    """Detecta objetos e retorna o output esperado

    Args:
        images_file: lista de arquivos de imagens para o detector procurar objetos
        output: o output que o detector deve retornar
        vars_output: dicionário com informações do output que o detector deve retornar

    Returns:
        O output passado para o detector

    """

    # carrega e pré-processa as imagens
    images = [await UploadImagePreprocessor.read_imagefile(image_file) for image_file in images_file]

    # instancia o estimador que será utilizado passando o modelo já carregado e informações úteis para predição
    estimator = Estimator(model_url=variaveis.model_url, infos=variaveis.info_uteis)

    # inicia a contagem do tempo de predição
    print('Predicting... ', end='')
    start_time = time.time()

    # utiliza o método predict do estimador (não é o método padrão para modelos Keras) passando as imagens, o output
    # esperado e algumas variáveis importantes
    response_object = estimator.predict(images, output=output, vars_output=vars_output)

    # finaliza a contagem do tempo de predição e exibe o tempo total em segundos
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))

    # O formato do response_object varia conforme o output passado
    return response_object

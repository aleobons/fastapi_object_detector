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

    # inicializa a lista de coordenadas
    coordenadas = []

    # percorre a lista de arquivos de imagens e executa a detecção em cada imagem, incluindo o resultado na lista de
    # coordenadas
    for image_file in images_file:
        # as coordenadas estão mapeadas com o índice 0
        output_coordenadas = variaveis.outputs.get('0', {})

        # executa a predição na imagem informando o output de coordenadas
        coordenadas.append(await execute(image_file, output=output_coordenadas))

    return coordenadas


@router.post("/crop", status_code=200)
async def post(images_file: UploadFile = File(...)):
    """Detecta objetos e retorna a recorte do objeto

        Args:
            images_file: arquivo de imagem para o detector procurar objetos

        Returns:
            A imagem recortada

        Raises:
            HTTPException: Um erro ocorre se não for detectado objeto.
        """

    # o crop está mapeado com o índice 1
    output_crop = variaveis.outputs.get('1', {})

    # executa a predição na imagem informando o output crop
    image = await execute(images_file, output=output_crop)

    # lida com a não detecção de nenhum objeto com um erro
    if image is None:
        raise HTTPException(status_code=406, detail="Nenhum objeto detectado na imagem.")

    # O StreamingResponse é utilizado para retornar a imagem já codificada em PNG
    return StreamingResponse(io.BytesIO(image), media_type="image/png")


@router.post("/vis_objects", status_code=200)
async def post(images_file: UploadFile = File(...)):
    """Detecta objetos e retorna as coordenadas

    Args:
        images_file: arquivo de imagem para o detector procurar objetos

    Returns:
        A imagem com os objetos anotados, os rótulos e as confianças

    """

    # a visualização dos objetos na imagem está mapeada com o índice 2
    output_vis_objects = variaveis.outputs.get('2', {})

    # executa a predição na imagem informando o output de visualização dos objetos
    image = await execute(images_file, output=output_vis_objects)

    # O StreamingResponse é utilizado para retornar a imagem já codificada em PNG
    return StreamingResponse(io.BytesIO(image), media_type="image/png")


async def execute(image_file, output):
    """Detecta objetos e retorna as coordenadas

    Args:
        image_file: arquivo de imagem para o detector procurar objetos
        output: dicionário com informações do output que o detector deve retornar

    Returns:
        A imagem com os objetos anotados, os rótulos e as confianças

    """

    # carrega e pré-processa a imagem
    image = await UploadImagePreprocessor.read_imagefile(image_file)

    # instancia o estimador que será utilizado passando o modelo já carregado e informações úteis para predição
    estimator = Estimator(model=variaveis.model, infos=variaveis.info_uteis)

    # inicia a contagem do tempo de predição
    print('Predicting... ', end='')
    start_time = time.time()

    # utiliza o método predict do estimador (não é o método padrão para modelos Keras) passando a imagem e o output
    # esperado
    response_object = estimator.predict(image, output=output)

    # finaliza a contagem do tempo de predição e exibe o tempo total em segundos
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))

    # O formato do response_object varia conforme o output passado
    return response_object

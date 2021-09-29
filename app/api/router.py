from fastapi import APIRouter, UploadFile, File
import io
from starlette.responses import StreamingResponse
from typing import List
from .preprocessors.uploadimagepreprocessor import UploadImagePreprocessor
from .estimators.objectdetector import ObjectDetector as Estimator
import api.global_variables as variaveis
import time

router = APIRouter()


@router.post("/coordenadas", status_code=200)
async def post(images_file: List[UploadFile] = File(...)):
    coordenadas = []
    for image_file in images_file:
        coordenadas.append(await execute(image_file, output=variaveis.outputs.get('0', {})))

    return coordenadas


@router.post("/crop", status_code=200)
async def post(images_file: UploadFile = File(...)):
    image = await execute(images_file, output=variaveis.outputs.get('1', {}))

    if image is not None and not type(image) == str:
        return StreamingResponse(io.BytesIO(image), media_type="image/png")
    else:
        return "no_license_plate"


@router.post("/vis_objects", status_code=200)
async def post(images_file: UploadFile = File(...)):
    image = await execute(images_file, output=variaveis.outputs.get('2', {}))
    return StreamingResponse(io.BytesIO(image), media_type="image/png")


async def execute(image_file, output):
    image = await UploadImagePreprocessor.read_imagefile(image_file)

    estimator = Estimator(model=variaveis.model, infos=variaveis.info_uteis)

    print('Predicting... ', end='')
    start_time = time.time()

    response_object = estimator.predict(image, output=output)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))

    return response_object

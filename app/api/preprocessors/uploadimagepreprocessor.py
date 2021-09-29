from PIL import Image
from PIL import UnidentifiedImageError
from io import BytesIO
from tensorflow.keras.preprocessing.image import img_to_array
from fastapi import HTTPException


class UploadImagePreprocessor:

    @staticmethod
    async def read_imagefile(upload_file):
        extension_image = upload_file.filename.split(".")[-1].lower() in ("jpg", "jpeg", "png")

        if not extension_image:
            raise HTTPException(status_code=415, detail="Unsupported file provided.")

        file = await upload_file.read()
        try:
            imagem = Image.open(BytesIO(file))
            imagem = img_to_array(imagem, dtype='uint8')
        except UnidentifiedImageError:
            raise HTTPException(status_code=415, detail="Invalid file.")

        return imagem

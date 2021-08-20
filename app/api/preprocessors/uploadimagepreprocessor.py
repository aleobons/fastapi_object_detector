import numpy as np
import cv2


class UploadImagePreprocessor:

    @staticmethod
    async def read_imagefile(upload_file):
        imagem = None

        extension_image = upload_file.filename.split(".")[-1] in ("jpg", "jpeg", "png")

        if not extension_image:
            return "Image must be jpg or png format!"

        file = await upload_file.read()
        np_file = np.fromstring(file, np.uint8)

        try:
            imagem = cv2.cvtColor(cv2.imdecode(np_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        except Exception as e:
            print(e)
        finally:
            if imagem is None:
                return "Invalid image"

            return imagem

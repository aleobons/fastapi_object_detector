import tensorflow as tf
from fastapi import HTTPException
import filetype


class ImageProcessor:
    """Classe que processa as imagens.

      Lê, decodifica

    """

    @staticmethod
    async def read_imagefile(upload_file):
        """Lê o arquivo de imagem.

        Lê e valida o arquivo de imagem

        Args:
          upload_file: arquivo de imagem

        Returns:
          Uma string com a leitura da imagem

        Raises:
          HTTPException: Um erro devido ao upload de um arquivo que não é imagem
        """

        # lê o arquivo
        input_image = await upload_file.read()

        # verifica qual o tipo de arquivo
        tipo_arquivo = filetype.guess(input_image)

        if tipo_arquivo is None:
            # caso não reconheça o tipo do arquivo, dispara um erro
            raise HTTPException(
                status_code=415, detail=f"Arquivo precisa ser uma imagem")
        elif tipo_arquivo.extension != 'jpg':
            # caso não seja jpeg dispara um erro
            raise HTTPException(
                status_code=415, detail=f"Imagem precisa ser jpeg. Foi enviado um arquivo {tipo_arquivo.extension}")

        # retorna a imagem lida
        return input_image

    @staticmethod
    def decode_image(input_image):
        return tf.image.decode_jpeg(input_image, channels=3).numpy()

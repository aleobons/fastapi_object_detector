import tensorflow as tf
from fastapi import HTTPException


class UploadImagePreprocessor:
    """Classe que lê os arquivos de imagens.

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

        # verifica se é uma imagem jpeg
        if not tf.io.is_jpeg(input_image, name=None):
            # caso negativo dispara um erro de arquivo inválido
            raise HTTPException(status_code=415, detail="Invalid file.")

        # retorna a imagem lida
        return input_image

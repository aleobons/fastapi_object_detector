import tensorflow as tf
from fastapi import HTTPException


class UploadImagePreprocessor:
    """Classe que pré-processa os arquivos de imagens.

    Possui apenas um método estático que lê o arquivo e converte em uma string decodificada
    """

    @staticmethod
    async def read_imagefile(upload_file):
        """Lê o arquivo de imagem.

        Lê, valida e converte o arquivo de imagem em uma string decodificada

        Args:
          upload_file: arquivo de imagem

        Returns:
          Uma string com a imagem decodificada

        Raises:
          HTTPException: Um erro devido ao upload de um arquivo que não é imagem
        """

        # primeiro verifica a extensão do arquivo, cuidando para aceitar extensão em caixa alta
        extension_image = upload_file.filename.split(".")[-1].lower() in ("jpg", "jpeg")

        # dispara um erro se a extensão não estiver na lista permitida
        if not extension_image:
            raise HTTPException(status_code=415, detail="Unsupported file provided.")

        # lê o arquivo e codifica o arquivo
        input_image = await upload_file.read()

        # verifica se é uma imagem jpeg
        if not tf.io.is_jpeg(input_image, name=None):
            # caso negativo dispara um erro de arquivo inválido
            raise HTTPException(status_code=415, detail="Invalid file.")

        # retorna a imagem lida
        return input_image

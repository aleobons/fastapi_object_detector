import tensorflow as tf
from fastapi import HTTPException


class UploadImagePreprocessor:
    """Classe que pré-processa os arquivos de imagens.

    Possui apenas um método estático que lê o arquivo e converte em array
    """

    @staticmethod
    async def read_imagefile(upload_file):
        """Lê o arquivo de imagem.

        Lê, valida e converte o arquivo de imagem em array

        Args:
          upload_file: arquivo de imagem

        Returns:
          Um array com a imagem

        Raises:
          HTTPException: Um erro devido ao upload de um arquivo que não é imagem
        """

        # primeiro verifica a extensão do arquivo, cuidando para aceitar extensão em caixa alta
        extension_image = upload_file.filename.split(".")[-1].lower() in ("jpg", "jpeg", "png")

        # dispara um erro se a extensão não estiver na lista permitida
        if not extension_image:
            raise HTTPException(status_code=415, detail="Unsupported file provided.")

        # lê o arquivo
        file = await upload_file.read()

        # tenta abrir o arquivo como imagem e converte para array
        try:
            imagem = tf.image.decode_jpeg(file, channels=3)
            imagem = imagem.numpy().tolist()

        except tf.errors.InvalidArgumentError:
            # se um erro de imagem não identificada for detectado, dispara um erro de arquivo inválido
            raise HTTPException(status_code=415, detail="Invalid file.")

        # retorna a imagem em formato de Numpy array
        return imagem

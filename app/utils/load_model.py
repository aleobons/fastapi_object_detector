import tensorflow as tf
from enum import Enum


class LoadModel:
    """ Classe para carregar o modelo

    Carrega tanto modelo KERAS como qualquer modelo Tensorflow

    Attributes:
        TypeModel(Enum): constantes que definem a forma de carregamento do modelo
        type_models: dicionário que define qual função utilizar para cada tipo de modelo
        type_model: tipo do modelo a ser carregado
        model_path: local onde o modelo está salvo

    """

    class TypeModel(Enum):
        MODEL_KERAS = 'KERAS'
        MODEL_TENSORFLOW = 'TENSORFLOW'

    def __init__(self, model_path: str, type_model: TypeModel):
        self.type_models = {
            LoadModel.TypeModel.MODEL_KERAS: self._load_model_keras,
            LoadModel.TypeModel.MODEL_TENSORFLOW: self._load_model_tensorflow
        }
        self.type_model = type_model
        self.model_path = model_path

    def carrega_modelo(self):
        return self.type_models.get(self.type_model)() # chama a função correspondente ao tipo do modelo

    def _load_model_keras(self):
        return tf.keras.models.load_model(self.model_path)

    def _load_model_tensorflow(self):
        return tf.saved_model.load(self.model_path)

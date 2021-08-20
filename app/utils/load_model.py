from tensorflow.keras.models import load_model
import tensorflow as tf


class LoadModel:
    MODEL_KERAS = 'KERAS'
    MODEL_TENSORFLOW = 'TENSORFLOW'

    def __init__(self, model_path, type_model):
        self.type_models = {
            LoadModel.MODEL_KERAS: self._load_model_keras,
            LoadModel.MODEL_TENSORFLOW: self._load_model_tensorflow()
        }

        self.type_model = type_model
        self.model_path = model_path

    def carrega_modelo(self):
        return self.type_models.get(self.type_model)()

    def _load_model_keras(self):
        return load_model(self.model_path)

    def _load_model_tensorflow(self):
        return tf.saved_model.load(self.model_path)

from locust import HttpUser, task, constant
import json
import tensorflow as tf
import os


class LoadTest(HttpUser):
    wait_time = constant(0)
    host = "http://localhost"

    @task
    def predict_placa_veiculo(self):
        image_path = "00011.jpg"

        raw = tf.io.read_file(os.path.join('/mnt/locust/files', image_path))
        image = tf.image.decode_jpeg(raw, channels=3)
        image = image[tf.newaxis, ...]
        image = image.numpy().tolist()

        # Create JSON to use in the request
        data = {"instances": image}

        headers = {"content-type": "application/json"}

        url = 'http://detector_placa_veiculos:8501/v1/models/detector_placa_veiculos/labels/stable:predict'

        self.client.request("POST", url, data=json.dumps(data), headers=headers)

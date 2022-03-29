from locust import HttpUser, task
import os


class LoadTest(HttpUser):
    host = "http://localhost"

    @task
    def predict_placa_veiculo(self):
        image_path = "00011.jpg"

        with open(os.path.join("files", image_path), "rb") as image:
            data = [("images_file", image)]

            url = "http://localhost:8401/detect_license_plate/crop"

            self.client.request("POST", url, files=data)

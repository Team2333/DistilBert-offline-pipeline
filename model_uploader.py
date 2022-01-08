from minio import Minio
from minio.error import S3Error
import datetime
import os
import requests



class Model_Uploader():
    def __init__(self, model_path):
        self.model_path = model_path
        self.bucket_name = "{model}"
        self.mc_client = "{client}"
        self.access_key = "{access_key}"
        self.secret_key = "{secret_key}"
        self.mc = Minio(self.mc_client, self.access_key, self.secret_key)
        current_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
        self.file_name = "distilbert_model.pt"

    def upload_2_minio(self):
        try:
            self.mc.fput_object(self.bucket_name, self.file_name, self.model_path)
            print(f"Sucessful in uploading model file: {self.file_name}")
            self.post()
        except Exception as e:
            print(f"Failure {e} in uploading model file: {self.file_name}")

    def post(self):
        
        url = f"https://api.covid-polygraph.ml/upload_model_weights/{self.bucket_name}/{self.file_name}"
        response = requests.post(url)
        if response.status_code == 200:
            print("Successfully post request")
        else:
            print(f"Error in posting request: Code {response.status_code}, Error Message: {response.text}")


if __name__ == "__main__":
    saved_weights_path = 'model/distilbert-base-cased_saved_weights.pt'
    mu = Model_Uploader(saved_weights_path)
    mu.upload_2_minio()
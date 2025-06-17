import requests
import numpy as np
import os
import cv2
import uuid
import json


class APIModel:
    def __init__(self, ip_address: str, port: int, model_path: str):
        self.model_path = model_path
        self.base_url = f"http://{ip_address}:{port}"
        self.host = ip_address
        self.port = port

    def _upload_model(self):
        """Upload the model to the API server."""
        with open(self.model_path, "rb") as f:
            files = {"file": (os.path.basename(self.model_path), f)}
            response = requests.post(f"{self.base_url}/upload-model/", files=files)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to upload model: {response.text}")

    def _upload_image(self, image: np.ndarray | list[np.ndarray]):
        """Upload the image to the API server."""

        def _process_image(img: np.ndarray):
            """Process the image for upload."""
            name = f"{uuid.uuid4()}.jpg"
            files = {"file": (name, cv2.imencode(".jpg", img)[1].tobytes())}
            response = requests.post(
                f"{self.base_url}/upload-img/{self.model}", files=files
            )
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Failed to upload model: {response.text}")

        if isinstance(image, list):
            results = []
            for img in image:
                results.append(_process_image(img))
            return results
        elif isinstance(image, np.ndarray):
            return _process_image(image)
        else:
            raise TypeError("Image must be a numpy array or a list of numpy arrays.")

    def _download_file(self, server_file_path: str, basef: str):
        """Download a file from the API server."""
        response = requests.get(
            f"{self.base_url}/download-image/{server_file_path}/{basef}", stream=True
        )
        if response.status_code != 200:
            raise Exception(f"Failed to download file: {response.text}")
        try:
            image_bytes = b"".join(
                chunk for chunk in response.iter_content(chunk_size=8192)
            )
            return image_bytes
        except Exception as e:
            raise Exception(f"Failed to extract bytes: {str(e)}")

    def __call__(self, image: np.ndarray | list[np.ndarray]):
        """Call the API with the image data."""
        if not hasattr(self, "model"):
            # Upload the model only once
            self.model = self._upload_model()

        def _get_result(data):
            summary_file_path = data["summary_file_path"]
            basef = data["baseF"]
            result = self._download_file(os.path.basename(summary_file_path), basef)
            result = result.decode("utf-8").replace("'", '"')
            return json.loads(result)

        data = self._upload_image(image)
        if isinstance(data, list):
            results = []
            for item in data:
                results.append(_get_result(item))
            return results
        else:
            return _get_result(data)

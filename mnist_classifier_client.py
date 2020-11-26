import requests
import json
import numpy as np
import base64
import pandas as pd


class LivenessDetectionClient:

    def __init__(self, host='0.0.0.0', port=8001):
        self._url = f'http://{host}:{port}/invocations'

    def predict(self, images):
        images = np.array(images)
        images_encoded = str(base64.b64encode(images), 'utf-8')
        data = pd.DataFrame(
            data=[images_encoded], columns=["image"]
        ).to_json(orient="split")
        r = requests.post(
            url=self._url,
            data=data,
            headers={"Content-Type": "application/json; format=pandas-split"},
        )
        assert r.status_code == 200
        return np.array(json.loads(r.text), dtype='float32')

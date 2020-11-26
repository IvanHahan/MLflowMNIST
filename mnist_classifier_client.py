import requests
import json
import numpy as np
import base64
import pandas as pd


class MNISTClassifierClient:

    def __init__(self, host='0.0.0.0', port=8001):
        self._url = f'http://{host}:{port}/invocations'

    def predict(self, images):
        images_encoded = [str(base64.b64encode(image), 'utf-8') for image in images]
        data = pd.DataFrame(
            data=images_encoded, columns=["image"]
        ).to_json(orient="split")
        r = requests.post(
            url=self._url,
            data=data,
            headers={"Content-Type": "application/json; format=pandas-split"},
        )
        assert r.status_code == 200
        return np.array(json.loads(r.text), dtype='float32')


if __name__ == '__main__':
    client = MNISTClassifierClient()
    x = np.random.randint(0, 255, (2, 1, 28, 28), dtype='uint8')  # batch x channel x height x width
    client.predict(x)

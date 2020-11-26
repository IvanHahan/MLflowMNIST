import cloudpickle
import torch
import torchvision
import pytorch_lightning

CONDA_ENV = {
        'name': 'mlflow-env',
        'channels': ['defaults', 'conda-forge', 'pytorch'],
        'dependencies': [
            'python=3.7.7',
            f'cloudpickle=={cloudpickle.__version__}',
            f'pytorch=={torch.__version__}',
            'mlflow',
            f'torchvision=={torchvision.__version__}',
            f'pytorch-lightning=={pytorch_lightning.__version__}',
        ]
    }
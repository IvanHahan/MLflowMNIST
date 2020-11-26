# Usage Guide

1. Install required dependencies
2. Start training `python mnist_classifier.py`.
 
    This will download MNIST dataset to the current directory, create
    `mlruns` folder for storing training experiments logs, checkpoints 
    and starts training.

3. Run `mlflow ui` to open mlfow dashboard and track training
history.
4. Run `mlflow models serve -m mlruns/0/<run id>/artifacts/model -h 0.0.0.0 -p 8001` to deploy
the trained model.

    The deployed model can be used using CURL or the implemented
    client inside `mnist_classifier_client.py`.

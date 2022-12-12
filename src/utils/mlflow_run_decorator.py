from functools import wraps

import mlflow

import sys
sys.path.append('src/')
from constants import PROJECT_EXPERIMENT_NAME

# mlflow run decorator from : https://www.sicara.fr/blog-technique/dvc-pipeline-runs-mlflow
def mlflow_run(wrapped_function):
    @wraps(wrapped_function)
    def wrapper(*args, **kwargs):
        mlflow.set_experiment(PROJECT_EXPERIMENT_NAME)
        with mlflow.start_run():  # recover parent run thanks to MLFLOW_RUN_ID env variable
            with mlflow.start_run(run_name=wrapped_function.__name__, nested=True):  # start child run
                return wrapped_function(*args, **kwargs)
    return wrapper

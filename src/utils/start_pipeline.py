import mlflow
import typer
import sys

sys.path.append('src/')
from constants import PROJECT_ROOT_PATH, PROJECT_EXPERIMENT_NAME

# code from : https://www.sicara.fr/blog-technique/dvc-pipeline-runs-mlflow
def start_pipeline(run_name):
    mlflow.set_experiment(PROJECT_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name):
        print(mlflow.active_run().info.run_id)
        mlflow.log_artifact(PROJECT_ROOT_PATH + '/' + "dvc.yaml")


if __name__ == "__main__":
    typer.run(start_pipeline)

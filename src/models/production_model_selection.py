import joblib
import dagshub
import mlflow
import argparse
from pprint import pprint
from train_model import read_params
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

def log_production_model(config_path):
    config = read_params(config_path)
    mlflow_config = config["mlflow_config"] 
    model_name = mlflow_config["registered_model_name"]
    model_dir = config["model_dir"]

    dagshub.init("Wild-Blueberry-MLOps", "ChiragChauhan4579", mlflow=True)
    mlflow.set_tracking_uri('https://dagshub.com/ChiragChauhan4579/Wild-Blueberry-MLOps.mlflow')
    MLFLOW_TRACKING_URI = "https://dagshub.com/ChiragChauhan4579/Wild-Blueberry-MLOps.mlflow"
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # experiment = client.get_experiment(experiment_id=0)
    # print("Name: {}".format(experiment.name))
    # print("Experiment_id: {}".format(experiment.experiment_id))
    # print("Artifact Location: {}".format(experiment.artifact_location))
    # print("Tags: {}".format(experiment.tags))
    # print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

    runs = client.search_runs(
        experiment_ids='0',
        max_results=5,
        order_by=["metrics.mean_absolute_error ASC"]
    )

    print(runs)

    for run in runs:
        print(f"run id: {run.info.run_id}, mae: {run.data.metrics['mean_absolute_error']:.4f}, \
            run params: {run.data.params}" )


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="C:/Users/Chirag/Desktop/Wild-Blueberry-MLOps/params.yaml")
    parsed_args = args.parse_args()
    data = log_production_model(config_path=parsed_args.config)
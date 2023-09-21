import joblib
import mlflow
import argparse
from pprint import pprint
from train_model import read_params
from mlflow.tracking import MlflowClient

def log_production_model(config_path):
    config = read_params(config_path)
    mlflow_config = config["mlflow_config"] 
    model_name = mlflow_config["registered_model_name"]
    model_dir = config["model_dir"]

    experiment = mlflow.get_experiment_by_name(mlflow_config["experiment_name"])

    runs = MlflowClient().search_runs(
        experiment_ids=experiment.experiment_id,
        filter_string="",
        max_results=1,
        order_by=["metrics.mae ASC"],
    )

    for run in runs:
        print(f"run id: {run.info.run_id}, mae: {run.data.metrics['mae']:.4f}, run params: {run.data.params}" )

    MlflowClient().transition_model_version_stage(
        name="xgboost_model_1", version=1, stage="Staging"
    )

    loaded_model = mlflow.pyfunc.load_model(f"C:/Users/Chirag/Desktop/Wild-Blueberry-MLOps/src/models/mlruns/{experiment.experiment_id}/{run.info.run_id}/artifacts/model")

    joblib.dump(loaded_model, model_dir)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="C:/Users/Chirag/Desktop/Wild-Blueberry-MLOps/params.yaml")
    parsed_args = args.parse_args()
    data = log_production_model(config_path=parsed_args.config)
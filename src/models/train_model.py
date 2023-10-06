import json
import yaml
import joblib
import mlflow
import argparse
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import *

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def evaluation(y_test,predictions):
    mae = mean_absolute_error(y_test,predictions)
    mse = mean_squared_error(y_test,predictions)
    return mae,mse 

def get_feat_and_target(df,target):
    x=df.drop(target,axis=1)
    y=df[[target]]
    return x,y    

def train_and_evaluate(config_path):
    config = read_params(config_path)
    train_data_path = config["processed_data_config"]["train_data_csv"]
    test_data_path = config["processed_data_config"]["test_data_csv"]
    target = config["raw_data_config"]["target"]
    max_depth=config["xgboost"]["max_depth"]
    n_estimators=config["xgboost"]["n_estimators"]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")
    train_x,train_y=get_feat_and_target(train,target)
    test_x,test_y=get_feat_and_target(test,target)

    mlflow_config = config["mlflow_config"]
    # remote_server_uri = mlflow_config["remote_server_uri"]

    # mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])
    experiment = mlflow.get_experiment_by_name(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=mlflow_config["run_name"],experiment_id=experiment.experiment_id) as mlops_run:
        model = XGBRegressor(max_depth=max_depth,n_estimators=n_estimators)
        model.fit(train_x, train_y)
        y_pred = model.predict(test_x)
        mae,mse = evaluation(test_y,y_pred)

        mlflow.log_param("max_depth",max_depth)
        mlflow.log_param("n_estimators", n_estimators)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
       
        mlflow.xgboost.log_model(
                model, 
                "model", 
                registered_model_name=mlflow_config["registered_model_name"])
        
        df_ref = train_x.copy(deep=True)
        df_curr = test_x.copy(deep=True)

        df_ref['target'] = train_y
        df_ref['prediction'] = train_y.values

        df_curr['target'] = test_y
        df_curr['prediction'] = y_pred

        data_drift_report = Report(metrics=[
            DataDriftPreset(),
        ])

        data_drift_report.run(reference_data=df_ref, current_data=df_curr)
        data_drift_report.save_html("C:/Users/Chirag/Desktop/Wild-Blueberry-MLOps/reports/report.html")
        mlflow.log_artifact("C:/Users/Chirag/Desktop/Wild-Blueberry-MLOps/reports/report.html")

 
if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="C:/Users/Chirag/Desktop/Wild-Blueberry-MLOps/params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)
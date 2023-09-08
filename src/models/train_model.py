import json
import dagshub
import yaml
import joblib
import mlflow
import argparse
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def read_params(config_path):
    """
    read parameters from the params.yaml file
    input: params.yaml location
    output: parameters as dictionary
    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def evaluation(y_test,predictions):
    """
    Evaluate the model performance based on the metrics
    input: actual and predicted values
    output: metrics
    """
    mae=mean_absolute_error(y_test,predictions)
    mse=mean_squared_error(y_test,predictions)
    rmse=np.sqrt(mse)
    r2=r2_score(y_test,predictions)
    return mae,mse,rmse,r2


def get_feat_and_target(df,target):
    """
    Get features and target variables seperately from given dataframe and target 
    input: dataframe and target column
    output: two dataframes for x and y 
    """
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

################### MLFLOW ###############################
    mlflow_config = config["mlflow_config"]
    dagshub.init("Wild-Blueberry-MLOps", "ChiragChauhan4579", mlflow=True)
    mlflow.set_tracking_uri('https://dagshub.com/ChiragChauhan4579/Wild-Blueberry-MLOps.mlflow')
    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=mlflow_config["run_name"]):
        model = XGBRegressor(max_depth=max_depth,n_estimators=n_estimators)
        model.fit(train_x, train_y)
        y_pred = model.predict(test_x)
        mae,mse,rmse,r2 = evaluation(test_y,y_pred)

        mlflow.log_param("max_depth",max_depth)
        mlflow.log_param("n_estimators", n_estimators)

        mlflow.log_metric("mean_absolute_error", mae)
        mlflow.log_metric("mean_squared_error", mse)
        mlflow.log_metric("root_mean_squared_error", rmse)
        mlflow.log_metric("r2_score", r2)

        mlflow.xgboost.log_model(
                model, 
                "model", 
                registered_model_name=mlflow_config["registered_model_name"])
 
if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="C:/Users/Chirag/Desktop/Wild-Blueberry-MLOps/params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)
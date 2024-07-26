import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
import pickle
import mlflow
import xgboost as xgb
import pathlib
from prefect import flow, task
from prefect_aws import S3Bucket
from datetime import date
from prefect.artifacts import create_markdown_artifact


@task(retries = 2, retry_delay_seconds = 3)
def load_data():
    s3_bucket_data = S3Bucket.load("my-s3-bucket")
    s3_bucket_data.download_folder_to_path(from_folder ="dataset", to_folder="dataset")
    df= pd.read_csv("dataset/flight_dataset.csv")
    categorical = ["Airline", "Source", "Destination"]
    numerical = ["Total_Stops","Duration_hours","Duration_min"] 
    df = df[categorical + numerical]
    df.Duration_hours = df.Duration_hours *60
    df["duration"] = df["Duration_hours"] + df["Duration_min"]
    return df


@task
def data_transformation(df):
    target = df["duration"].values
    df = df.drop(["Duration_hours", "Duration_min", "duration"], axis = 1)
    df = df.to_dict(orient = "records")
    dv = DictVectorizer()
    data_df = dv.fit_transform(df)
    train_df = data_df[:8000]
    test_df = data_df[8000:]
    y_train =  target[:8000]
    y_test = target[8000:]
    return train_df, test_df, y_train, y_test, dv

@task(log_prints = True)
def train_model(
    X_train,
    X_val,
    y_train,
    y_val,
    dv
) -> None:
    """train a model with best hyperparams and write everything out"""

    with mlflow.start_run():
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            "learning_rate": 0.16968989909872087,
            "max_depth": 25,
            "min_child_weight": 5.591678840975327,
            "objective": "reg:linear",
            "reg_alpha": 0.11973660565878817,
            "reg_lambda": 0.020803099001553724,
            "seed": 42,
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, "validation")],
            early_stopping_rounds=5,
        )

        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred,squared=False)
        mlflow.log_metric("rmse", rmse)

        pathlib.Path("models").mkdir(exist_ok=True)
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        markdown__rmse_report = f"""# RMSE Report

        ## Summary

        Duration Prediction 

        ## RMSE XGBoost Model

        | Region    | RMSE |
        |:----------|-------:|
        | {date.today()} | {rmse:.2f} |
        """

        create_markdown_artifact(
            key="duration-model-report", markdown=markdown__rmse_report
        )

        

    return None




@flow
def main_flow_s3():

    mlflow.set_tracking_uri("mlflow.db")
    
    mlflow.set_experiment("Flight prediction time")
    

    data = load_data()
    X_train, X_test, y_train, y_test, dv = data_transformation(data)

    train_model(X_train, X_test, y_train, y_test, dv )


if __name__ == "__main__":
    main_flow_s3()
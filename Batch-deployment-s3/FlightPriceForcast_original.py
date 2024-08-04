# %%
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
import os
import pickle
from sklearn.pipeline import make_pipeline
import mlflow
import uuid


# %%
output_file = 'prediction.csv'
input_file = "dataset/flight_dataset.csv"

RUN_ID="16aa4ec2992e4def9a579a09802f7d54"


# %%
def load_data(input_file):
    df= pd.read_csv(input_file)
    categorical = ["Airline", "Source", "Destination"]
    numerical = ["Total_Stops","Duration_hours","Duration_min"]
    df = df[categorical + numerical]
    ride_list = generate_id(df)
    df['ride_id'] = ride_list
    return df

# %%
def preprocess_data_to_dict(df):
    df.Duration_hours = df.Duration_hours *60
    df["duration"] = df["Duration_hours"] + df["Duration_min"]
    target = df["duration"].values
    df = df.drop(["Duration_hours", "Duration_min", "duration"], axis = 1)
    df = df.to_dict(orient = "records")
    return df

# %%
def generate_id(df):
    ride_list = []
    for i in range(len(df)):
        ride_id = str(uuid.uuid4())
        ride_list.append(ride_id)
    return ride_list

# %%
def load_model(RUN_ID):
    logged_model = f's3://mlop-zoomcamp-adebayo/3/{RUN_ID}/artifacts/model'
    # Load model as a PyFuncModel.
    model = mlflow.pyfunc.load_model(logged_model)
    return model

# %%
def apply_model(RUN_ID, output_file, input_file):
    data = load_data(input_file)
    dict_df = preprocess_data_to_dict(data)
    model = load_model(RUN_ID)

    pred = model.predict(dict_df)
    df_result  = pd.DataFrame()

    df_result['ride_id'] =  data['ride_id'] 
    df_result['Airline'] = data['Airline']
    df_result['Source'] = data['Source']
    df_result['Destination'] = data['Destination']
    df_result['Total_Stops'] = data['Total_Stops']
    df_result['duration'] = data['duration']
    df_result['predicted_duration'] = pred
    df_result['Loss'] = df_result['predicted_duration'] - df_result['duration']
    df_result['model_version'] = RUN_ID

    df_result.to_csv(output_file)


# %%
def run():
    apply_model(
        RUN_ID, 
        output_file, 
        input_file)



if __name__ == "__main__":
    run()




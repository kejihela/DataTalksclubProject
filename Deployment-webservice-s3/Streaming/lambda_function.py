import json
import os
import base64
import boto3
import mlflow

    
boto3_kinesis = boto3.client("kinesis") 
PREDICTIONS_STREAM_NAME = os.getenv('PREDICTIONS_STREAM_NAME', 'kinesis_mlzoomcamp')

TEST_RUN = os.getenv('TEST_RUN', 'False') == 'True'
RUN_ID="16aa4ec2992e4def9a579a09802f7d54"

logged_model = f's3://mlop-zoomcamp-adebayo/3/{RUN_ID}/artifacts/model'
    # Load model as a PyFuncModel.
model = mlflow.pyfunc.load_model(logged_model)
    


def preprocess_data_to_dict(df):
    '''{
    "ride": {
        "Airline": 'IndiGo',
        "Source": 'Banglore',
        "Destination": New Delhi,
        "Total_Stops": 0
    }, 
    "ride_id": 123
}'''
    df = df.to_dict(orient = "records")
    return df

def predict(features):
    pred = model.predict(features)
    return float(pred[0])

def lambda_handler(event, context):
    predictions_events = []  
    if not TEST_RUN:
        for records in event['Records']:
            data_encoded = records['kinesis']['data']
            decoded_data = base64.b64decode(data_encoded).decode('utf-8')
            print(decoded_data)
            ride_events = json.loads(decoded_data)
            
    
        
    ride = ride_events['ride']
    ride_id = ride_events['ride_id']
    
    
    print(ride)
    prediction = predict(ride)
    

    prediction_event = {
        'model':'Best model',
        'version':'123',
        'prediction': {
            'ride_duration' : prediction,
            'ride_id':ride_id
            
        }
        
    }
        
      

    boto3_kinesis.put_record(
        StreamName=PREDICTIONS_STREAM_NAME,
        Data =json.dumps(prediction_event),
        PartitionKey=str(ride_id)
        )
    
    predictions_events.append(prediction_event) 

   
    return {
        'prediction': prediction_event
          }

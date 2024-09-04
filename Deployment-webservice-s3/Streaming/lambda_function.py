import json
import os
import base64
import boto3
def predict(feature):
    return 10.0
    
boto3_kinesis = boto3.client("kinesis") 
PREDICTIONS_STREAM_NAME = os.getenv('PREDICTIONS_STREAM_NAME', 'kinesis_mlzoomcamp')


def lambda_handler(event, context):
    predictions_events = []  
    
    for records in event['Records']:
        data_encoded = records['kinesis']['data']
        decoded_data = base64.b64decode(data_encoded).decode('utf-8')
        print(decoded_data)
        ride_events = json.loads(decoded_data)
        
    
        
    ride = ride_events['ride']
    ride_id = ride_events['ride_id']
    
    
    
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

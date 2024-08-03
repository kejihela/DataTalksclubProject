import pickle
from flask import Flask, jsonify, request
import mlflow



#mlflow.set_tracking_uri("http://127.0.0.1:5000")

#mlflow.set_experiment("S3-deployment-2")
RUN_ID="16aa4ec2992e4def9a579a09802f7d54"
logged_model = f's3://mlop-zoomcamp-adebayo/3/{RUN_ID}/artifacts/model'


# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)


app = Flask("Flight-prediction")


def predict(features):
    pred = loaded_model.predict(features)
    return float(pred)

@app.route('/predict', methods=["POST"])
def predict_endpoint():
    data = request.get_json()

    pred = predict(data)

    result =  {
        'duration': pred,
        'Run_id' : RUN_ID
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port='9696' )
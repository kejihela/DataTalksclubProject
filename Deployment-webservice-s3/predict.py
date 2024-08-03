import pickle
from flask import Flask, jsonify, request
import mlflow


logged_model = 'runs:/224f5aac05a64508919581fd1889e2ba/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)



def predict(features):
    data = dv.transform(features)
    pred = loaded_model.predict(data)
    return float(pred)

@app.route('/predict', methods=["POST"])
def predict_endpoint():
    data = request.get_json()

    pred = predict(data)

    result =  {
        'duration': pred
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port='9696' )
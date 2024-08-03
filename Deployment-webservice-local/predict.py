import pickle
from flask import Flask, jsonify, request

with open('model/artifacts.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)
app = Flask("flight-prediction") 

def predict(features):
    data = dv.transform(features)
    pred = model.predict(data)
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
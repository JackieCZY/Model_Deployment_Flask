from crypt import methods
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

#Create Flask App
app = Flask(__name__)

#Load pickle model
model = pickle.load(open("model.pkl","rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    return render_template("index.html", prediction_text = "Illness (1 Yes/0 No) {}".format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
    

if __name__== "__main__":
    app.run(debug = True)
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("Home.html")

@flask_app.route("/PredictDeformity")
def PredictDeformity():
     return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    feature_vector= [[float(x) for x in request.form.values()]]
    e6=feature_vector[0][0]+feature_vector[0][2]+feature_vector[0][4]
    e7=sum(feature_vector[0])
    e8=feature_vector[0][2]/2+feature_vector[0][3]+feature_vector[0][4]
    e9=feature_vector[0][2]/2+feature_vector[0][1]+feature_vector[0][0]
    feature_vector[0].append(e6)
    feature_vector[0].append(e7)
    feature_vector[0].append(e8)
    feature_vector[0].append(e9)
    prediction = model.predict(feature_vector)
    
    pred=int(prediction)
    return  render_template("index.html", prediction_text = ["You have a normal feet ","you have a flatfoot"][pred])
if __name__ == "__main__":
    flask_app.run(debug=True)


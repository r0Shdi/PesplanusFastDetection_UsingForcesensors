from flask import Flask, request, jsonify, render_template,redirect
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("/Users/roshdyhamdy/Desktop/FinalProject/website/model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("Home.html")


@flask_app.route("/predict", methods = ["POST"])
def predict():
    feature_vector= [float(x) for x in request.form.values()]
    WeightKG=feature_vector[0]
    FV=[[x*0.00311708261/WeightKG for x in feature_vector[1:]]]
    e6=FV[0][0]+FV[0][2]+FV[0][4]
    e7=sum(FV[0])
    e8=FV[0][2]/2+FV[0][3]+FV[0][4]
    e9=FV[0][2]/2+FV[0][1]+FV[0][0]
    FV[0].append(e6)
    FV[0].append(e7)
    FV[0].append(e8)
    FV[0].append(e9)
    prediction = model.predict(FV)
    
    pred=int(prediction)
    return  redirect(["/#normalfoot","/#flatfoot"][pred])
if __name__ == "__main__":
    flask_app.run(debug=True)


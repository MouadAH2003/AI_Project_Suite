from flask import Flask , request , jsonify
import joblib



app = Flask(__name__)
model = joblib.load("dates_fruit_classifier.pkl")

@app.route('/')
def home():
    return "Welcome to the Dates Fruit Classifier API"


@app.route("/predict" , methods=["POST"])
def predict():
    prediction = model.predict([[143957	,1452.2620	,523.1368	,352.3305,	0.7392,	428.1259,	0.9844,146236,	0.7746,	1.4848	,0.8577	,0.8184	,0.0036	,0.0024	,0.6697	,0.9944	,154.6815	,150.4548	,125.8076	,34.9626	,29.1245	,30.8160	,-1.3092	,-0.8756	,-0.3565	,4.1478	,3.4278	,2.4134	,-36987568128	,-34249711616	,-23752157184	,77.3446	,75.2287	,62.9057]])
    
    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(debug=True)
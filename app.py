import numpy as np
from flask import Flask, request, jsonify, render_template
from keras.utils import to_categorical
import pickle

app = Flask(__name__)
model = pickle.load(open("model_iris.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    klasisfikasi_bunga = ['SETOSA','VERSICOLOR','VIRGINICA']
    features = [np.array(float_features)]
    prediction = model.predict(features)
    hasil = klasisfikasi_bunga[prediction[0]]
    return render_template("index.html", prediction_text = "{}".format(hasil))

if __name__ == "__main__":
    app.run(debug=True)
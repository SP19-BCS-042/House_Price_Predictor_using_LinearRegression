from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
data = pd.read_csv("Cleaned_data.csv")
pipe = pickle.load(open("RidgeModel.pkl", "rb"))


@app.route("/")
def index():
    locations = sorted(data["location"].unique())
    return render_template('index.html', locations=locations)


@app.route("/predict", methods=["POST"])
def predict():


    location = request.form.get("location")
    bhk = request.form.get("bhk")
    if bhk is not None:
        bhk = float(bhk)
    else:
        bhk = 0.0  # or some other default value

    bath = request.form.get("bath")
    if bath is not None:
        bath = float(bath)
    else:
        bath = 0.0  # or some other default value

    sqft = request.form.get("total_sqft")
    if sqft is not None:
        sqft = float(sqft)
    else:
        sqft = 0.0  # or some other default value

    print(location,bhk,bath, sqft)
    input = pd.DataFrame([[location, sqft, bath, bhk]], columns=["location", "total_sqft", "bath", "bhk"])
    prediction = pipe.predict(input)[0] * 100000

    return str(np.round(prediction,2))


if __name__ == "__main__":
    app.run(debug=True, port=5000)

import numpy as np
import pickle
import joblib
import matplotlib
import matplotlib.pyplot as plt
import time
import pandas
import pandas as pd
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

model = pickle.load(open("C:/Users/rsana/Machine Learning Approach For Predicting The Rainfall/rainfall.pkl", 'rb')) 
scale = pickle.load(open("C:/Users/rsana/Machine Learning Approach For Predicting The Rainfall/scale.pkl",'rb'))


@app.route('/') #route to display the home page
def home(): 
    return render_template('index.html') #rendering the home page

@app.route('/predict', methods=["POST", "GET"])# route to show the predictions in a web UI

def predict():
    #reading the inputs given by the user I
    input_feature = [x for x in request.form.values()] 
    features_values = [np.array(input_feature)]
    names = [['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed',
        'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressuregam', 'Pressure3pm', 'Temp9am', 'Temp3pm', 'RainToday',
        'WindGustDir', 'WindDir9am', 'WindDir3pm', 'year', 'month', 'day']]
    data = pd.DataFrame(features_values, columns=names)
    data = data.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric (including strings to NaN)
    data.fillna(0, inplace=True) 
    
    data = scale.fit_transform(data)
    data = pandas.DataFrame(data, columns=names)
    # predictions using the loaded model file
    prediction = model.predict(data)
    pred_prob = model.predict_proba(data) 

    # Convert prediction to string
    prediction_str = 'Yes' if prediction[0] == 1 else 'No'

    print(prediction_str)
    if prediction_str == "Yes":  # Use prediction_str instead of prediction
        return render_template("chance.html") 
    else:
        return render_template("nochance.html")

if __name__ == "__main__":
    app.run(debug=True)
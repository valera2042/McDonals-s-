# -*- coding: utf-8 -*-
import flask
import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Use pickle to load in the pre-trained model

app = flask.Flask(__name__)
with open(f'model/Forest_ready.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='back')

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('mainm.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        Fat = flask.request.form['Fat']
        Sugar = flask.request.form['Sugar']

        # Make DataFrame for model
        input_variables = pd.DataFrame([[Fat, Sugar]],
                                       columns=['Fat', 'Sugar'],
                                       dtype=int,
                                       index=['input'])

        # Get the model's prediction
        prediction = model.predict(input_variables)[0]
    
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('mainm.html',
                                     original_input={'Fat':Fat,
                                                     'Sugar':Sugar},
                                     result=prediction,
                                     )

if __name__ == '__main__':
    app.run(debug=True)

    
    



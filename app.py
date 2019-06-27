from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap 
import pandas as pd 
import numpy as np 
import pickle

# ML Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib


app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():
	
	# Loading the ML Model. Which was previously created, trained and saved.
	tree_regressor = pickle.load(open("models/tree_model.pickle","rb"))

	s=2
	# Receives the input query from form
	if request.method == 'POST':
		# Receives the price input:
		pricequery = request.form['pricequery']
		# Receives the quality of the shelf location as an input:
		slq = request.form['shelvelocquery']
		shelvelocquery=0
		if slq=='Medium':
			shelvelocquery=1
		if slq=='Good':
			shelvelocquery=2 
		# Receives the local advertising budget (in thousands) as input:
		advertisingquery = request.form['advertisingquery']

		# An array is created with the input, in order to allow usage of the model:
		inputArray = [[pricequery,shelvelocquery,advertisingquery]]
		my_prediction = tree_regressor.predict(inputArray)
		my_prediction=my_prediction[0]

		# The prediction with other variables are returned along with the template 'index.html'
	return render_template('index.html',result = my_prediction,priceholder=pricequery,shelveLoc=shelvelocquery,adholder=advertisingquery)


if __name__ == '__main__':
	app.run(debug=True)
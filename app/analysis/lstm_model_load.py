import 	os
import 	glob
import 	tensorflow 	as 		tf
from 	tensorflow 	import 	keras
import 	pandas		as 		pd
import	numpy		as		np

from sklearn.preprocessing import  MinMaxScaler # This Scaler removes the median and scales the data according to the quantile range to normalize the price data 


import math # Mathematical functions 
import numpy as np # Fundamental package for scientific computing with Python
import pandas as pd # Additional functions for analysing and manipulating data
import time
import os
import os.path
from datetime import date, timedelta, datetime # Date Functions
#from pandas.plotting import register_matplotlib_converters # This function adds plotting functions for calender dates
#import matplotlib.pyplot as plt # Important package for visualization - we use this to plot the market data
#import matplotlib.dates as mdates # Formatting dates
import tensorflow as tf
from tensorflow import keras



from sklearn.metrics import mean_absolute_error, mean_squared_error # Packages for measuring model performance / errors
from tensorflow.keras.models import Sequential # Deep learning library, used for neural networks
from tensorflow.keras.layers import LSTM, Dense, Dropout # Deep learning classes for recurrent and regular densely-connected layers
from tensorflow.keras.callbacks import EarlyStopping # EarlyStopping during model training
from sklearn.preprocessing import RobustScaler, MinMaxScaler # This Scaler removes the median and scales the data according to the quantile range to normalize the price data 
#import seaborn as sns
 
# To remove the scientific notation from numpy arrays
np.set_printoptions(suppress=True)



def getFullPath( modulePath, subModulePath):
	return os.path.join( os.getcwd(),"app", modulePath, subModulePath)

def getCreatedModelSymbols():
	
	dirPath 		= getFullPath( 	modulePath	  = "analysis",
									subModulePath = "models/multivariate_price_data_only"
					)
	
	modelFilenames 	= \
		[os.path.basename(f) for f in glob.glob(os.path.join(dirPath,"*.h5"))]
	return [s.split("_")[1] for s in modelFilenames]

def getCreatedModelSymbolsNames():
	
	dirPath 		= getFullPath( 	modulePath	  = "analysis",
									subModulePath = "historical_data"
					)
	
	symbolList 		= getCreatedModelSymbols()
	
	df 				= pd.read_csv(	os.path.join(dirPath,"listed-companies.csv"),
									usecols=["Name", "Instrument_Code"] 
					)
					
	mDict 			= df.set_index("Instrument_Code").T.to_dict("list")
	
	# Create new dictionary based on created models
	d 				= {s: mDict[s][0] for s in symbolList }
	
	# Sort the list
	return { key:d[key] for key in sorted(d.keys())}

def getDataScalers(df_w_features):
	# Get the number of rows in the data
	nrows			 = df_w_features.shape[0]

	# Convert the data to numpy values
	np_data_unscaled = np.array(df_w_features)
	np_data 		 = np.reshape(np_data_unscaled, (nrows, -1))
	
	# Transform the data by scaling each feature to a range between 0 and 1
	scaler 			 = MinMaxScaler()
	np_data_scaled   = scaler.fit_transform(np_data_unscaled)

	# Creating a separate scaler that works on a single column for scaling predictions
	scaler_pred		 = MinMaxScaler()
	df_filtered_ext  = df_w_features.copy()
	df_filtered_ext['Prediction'] = df_filtered_ext['Close_Price']
	
	df_Close 		 = pd.DataFrame(df_filtered_ext['Close_Price'])
	np_Close_scaled  = scaler_pred.fit_transform(df_Close)
	
	return (scaler,  scaler_pred )

def create_LSTM_mv_model2(StockData, symbol_str, features, epochs, lstm_model):
	
	def LSTM_mv_scaleData(StockData):
							
		# Replace nan with zero
		df_m = StockData.fillna(0)

		# create index with date
		df_m = df_m.set_index("Date")

		# Indexing Batches
		train_df = df_m.sort_values(by=['Date']).copy()

		# We save a copy of the dates index, before we reset it to numbers
		date_index = train_df.index

		# We reset the index, so we can convert the date-index to a number-index
		train_df = train_df.reset_index(drop=True).copy()
		
		
		# FEATURES = [#'Today_High', 
					# #'Today_Low', 
					# 'Close_Price', 
					# #'Volume_non_block', 
					# #"no_days_not_traded_since_last_traded", 
					# "sentiment"
					# #"is_regarding_financial_report", 
					# #"is_sold_pur_shares"
			   # ]

		# Create the dataset with features and filter the data to the list of FEATURES
		data = pd.DataFrame(train_df)
		data_filtered = data[features]

		# We add a prediction column and set dummy values to prepare the data for scaling
		data_filtered_ext = data_filtered.copy()
		data_filtered_ext['Prediction'] = data_filtered_ext['Close_Price']
		
		# Get the number of rows in the data
		nrows = data_filtered.shape[0]

		# Convert the data to numpy values
		np_data_unscaled = np.array(data_filtered)
		np_data = np.reshape(np_data_unscaled, (nrows, -1))
		print(np_data.shape)

		# Transform the data by scaling each feature to a range between 0 and 1
		scaler = MinMaxScaler()
		np_data_scaled = scaler.fit_transform(np_data_unscaled)

		# Creating a separate scaler that works on a single column for scaling predictions
		scaler_pred = MinMaxScaler()
		df_Close = pd.DataFrame(data_filtered_ext['Close_Price'])
		np_Close_scaled = scaler_pred.fit_transform(df_Close)
		
		
		return (np_data_scaled,
				data_filtered,
				data.columns.get_loc("Close_Price"),
				scaler_pred
		)

	np_data_scaled,	\
	data_filtered,  \
	index_Close,    \
	scaler_pred		= LSTM_mv_scaleData(StockData)
	
	def LSTM_mv_DataPreparation (sequence_length):

		# sequence_length: Next few day's Price Prediction is based on the no of previous day's prices
		#			 that is set here.

		# Split the training data into train and train data sets
		# As a first step, we get the number of rows to train the model on 80% of the data 
		train_data_len = math.ceil(np_data_scaled.shape[0] * 0.8)

		# Create the training and test data
		train_data = np_data_scaled[0:train_data_len, :]
		test_data  = np_data_scaled[train_data_len - sequence_length:, :]

		# The RNN needs data with the format of [samples, time steps, features]
		# Here, we create N samples, sequence_length time steps per sample, and the no of features
		def partition_dataset(sequence_length, data):
			x, y = [], []
			data_len = data.shape[0]
			for i in range(sequence_length, data_len):
				x.append(data[i-sequence_length:i,:]) #contains sequence_length values 0-sequence_length * columns
				y.append(data[i, index_Close]) #contains the prediction values for validation
			
			# Convert the x and y to numpy arrays
			x = np.array(x)
			y = np.array(y)
			return x, y

		# Generate training data and test data
		x_train, y_train = partition_dataset(sequence_length, train_data)
		x_test, y_test   = partition_dataset(sequence_length, test_data)	
		

		return ( x_train, x_test,
				 y_train, y_test
		)	
	
	x_train, x_test, \
	y_train, y_test = LSTM_mv_DataPreparation( 	sequence_length = 50 )
	
	
	def LSTM_mv_ModelEvaluation():

		# Get the predicted values
		y_pred_scaled = lstm_model.predict(x_test)

		# Unscale the predicted values
		y_pred = scaler_pred.inverse_transform(y_pred_scaled)
		y_test_unscaled = scaler_pred.inverse_transform(y_test.reshape(-1, 1))

		# Mean Absolute Error (MAE)
		MAE = mean_absolute_error(y_test_unscaled, y_pred)
		print(f'Median Absolute Error (MAE): {np.round(MAE, 2)}')

		# Median Absolute Error (MedAE)
		MEDAE = np.median(abs(y_pred - y_test_unscaled))
		print(f'Median Absolute Error (MedAE): {np.round(MEDAE, 2)}')

		# Mean Squared Error (MSE)
		MSE = np.square(np.subtract(y_pred, y_test_unscaled)).mean()
		print('Mean Squared Error (MSE): ' + str(np.round(MSE, 2)))

		# Root Mean Squarred Error (RMSE) 
		RMSE = np.sqrt(np.mean(np.square(y_pred - y_test_unscaled)))
		print('Root Mean Squared Error (RMSE): ' + str(np.round(RMSE, 2)))

		# Mean Absolute Percentage Error (MAPE)
		MAPE = np.mean((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled))) * 100
		print(f'Mean Absolute Percentage Error (MAPE): {np.round(MAPE, 2)} %')

		# Median Absolute Percentage Error (MDAPE)
		MDAPE = np.median((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled)) ) * 100
		print(f'Median Absolute Percentage Error (MDAPE): {np.round(MDAPE, 2)} %')

		return (MAE, MEDAE, MSE, RMSE, MAPE, MDAPE)

	MAE, MEDAE, MSE, RMSE, MAPE, MDAPE = LSTM_mv_ModelEvaluation()
	
	
	return

def lstm_model_load (symbol, isSentimentModel):

	epochs = 25
	features_name_lst = ["Today_High","Today_Low","Close_Price", "Volume_non_block"]

	
	#### Determine model folder and create filename string for model
	if isSentimentModel == "True":
		subPath 	 = "models/multivariate_price_data_sentiments"
		featureCount = 5
		features_name_lst.append("sentiment")
	else:
		subPath = "models/multivariate_price_data_only"
		featureCount = 4
	
	dirPath 		= getFullPath( 	modulePath	  = "analysis",
									subModulePath = subPath
					)
	
	model_fn = '_'.join(["mv", symbol, str(featureCount), str(epochs),"lstm_model.h5"])
	
	
	##### Load models
	multi_lstm_model 	= tf.keras.models.load_model(
								os.path.join(dirPath, model_fn)
						)
	print(multi_lstm_model.summary())
	#### Load dataset
	df 	= pd.read_csv(	os.path.join('./app/analysis/historical_data/dataset',
								'prices_sentiment_'+symbol+'.csv'),
								infer_datetime_format=True, 
								sep=','
		)
	
	# Replace nan with zero
	df = df.fillna(0)
	
	#### Define DataScalers
	scaler, scaler_pred = getDataScalers(df[features_name_lst])
	
	sequence_length = 50
	
	df_temp = df[-sequence_length:]
	new_df 	= df_temp.filter(features_name_lst)

	N 		= sequence_length

	# Get the last N day closing price values and scale the data to be values between 0 and 1
	last_N_days 		= new_df[-sequence_length:].values
	last_N_days_scaled  = scaler.transform(last_N_days)

	# Create an empty list and Append past N days
	X_test_new = []
	X_test_new.append(last_N_days_scaled)
	
	# Convert the X_test data set to a numpy array and reshape the data
	pred_price_scaled 	= multi_lstm_model.predict(np.array(X_test_new))
	pred_price_unscaled = scaler_pred.inverse_transform(pred_price_scaled.reshape(-1, 1))
	
	print(pred_price_unscaled)
	
	# Print last price and predicted price for the next day
	predicted_Price = np.round(pred_price_unscaled.ravel()[0], 2)

	return predicted_Price
						
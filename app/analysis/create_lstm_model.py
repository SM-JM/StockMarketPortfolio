########################################################################
## Author: Shane Miller (620042891)
## LSTM Model adapted from: https://thinkingneuron.com/predicting-stock-prices-using-deep-learning-lstm-model-in-python/
##
########################################################################

import 	pandas 			as pd
import 	numpy 			as np
import 	os
import 	time
from 	datetime 		import datetime

# Feature Scaling for fast training of neural networks
from 	sklearn.preprocessing 	import StandardScaler, MinMaxScaler
# Importing the Keras libraries and packages
from 	tensorflow.keras.models	import Sequential
from 	tensorflow.keras.layers	import Dense
from 	tensorflow.keras.layers	import LSTM
 

# To remove the scientific notation from numpy arrays
np.set_printoptions(suppress=True)

def scaleData(StockData):
						
	FullData			= StockData[['Close_Price']].values
	 
	# Choosing between Standardization or normalization
	sc					= MinMaxScaler()
	DataScaler 			= sc.fit(FullData)
	scaled				= DataScaler.transform(FullData)
	
	return scaled.reshape(scaled.shape[0],)

def LSTM_DataPreparation (Data, TimeSteps, FutureTimeSteps ):

	# Data: 	 Stock data to scale and prepare
	# TimeSteps: Next few day's Price Prediction is based on the no of previous day's prices
	#			 that is set here.
	# FutureTimeSteps: How many days in future you want to predict the prices

	X 					= scaleData(Data) # Scale and convert to one dimensional array
	
	# Split into samples
	X_samples 			= list()
	y_samples 			= list()
	 
	NumerOfRows 		= len(X)
	 
	# Iterate thru the values to create combinations
	for i in range(TimeSteps , NumerOfRows-FutureTimeSteps , 1):
		x_sample 		= X[i-TimeSteps:i]
		y_sample 		= X[i:i+FutureTimeSteps]
		X_samples.append(x_sample)
		y_samples.append(y_sample)
	 
	 
	# Reshape the Input as a 3D (samples, Time Steps, Features)
	X_data				= np.array(X_samples)
	X_data				= X_data.reshape(X_data.shape[0],X_data.shape[1], 1)

	# We do not reshape y as a 3D data  as it is supposed to be a single column only
	y_data				= np.array(y_samples)

	return (X_data, y_data)

def LSTM_sampling (x, y, TestingRecords):
	# x 			: Input variable -> Price data used to predict target
	# y 			: Target variable/price to be predicted
	# TestingRecords: The number of days to use to test model

	# Splitting the data into train and test
	X_train				= x[:-TestingRecords]
	X_test				= x[-TestingRecords:]
	
	y_train				= y[:-TestingRecords]
	y_test				= y[-TestingRecords:]
	
	return ( X_train, X_test,
			 y_train, y_test
	)
	
def LSTM_MultiModel(X_train, y_train, TimeSteps, TotalFeatures, FutureTimeSteps):

	# Initialising the RNN
	regressor = Sequential()
	 
	# Adding the First input hidden layer and the LSTM layer
	# return_sequences = True, means the output of every time step to be shared 
	# with hidden next layer
	regressor.add(
		LSTM(
			units = 10, 
			activation = 'relu', 
			input_shape = (TimeSteps, TotalFeatures), 
			return_sequences=True
		)
	)
	 
	# Adding the Second hidden layer and the LSTM layer
	regressor.add(
		LSTM(
			units = 5, 
			activation = 'relu', 
			input_shape = (TimeSteps, TotalFeatures), 
			return_sequences=True
		)
	)
	 
	# Adding the Third hidden layer and the LSTM layer
	regressor.add(
		LSTM(
			units = 5, 
			activation = 'relu', 
			return_sequences=False 
		)
	)
	 
	# Adding the output layer
	# Notice the number of neurons in the dense layer is now the number of future time steps 
	# Based on the number of future days we want to predict
	regressor.add(Dense(units = FutureTimeSteps))
	 
	# Compiling the RNN
	regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
	 
	 
	##### Measuring the time taken by the model to train
	
	StartTime			= time.time()
	# Fitting the RNN to the Training set
	regressor.fit(X_train, y_train, batch_size = 5, epochs = 100)
	EndTime				= time.time()
	
	#####
	
	
	# Return the model and the total time taken to run the model
	return ( regressor, 
			 "Total Time Taken: {} minutes.".format(round((EndTime-StartTime)/60)) 
	 )

def create_lstm_model(StockData, symbol_str, pNoFuturePredictions):
	
	# Prepare the data into the inputs X_data (last 10 day prices) and 
	# y_data (11th day or targe price to be predicted).
	X_data, y_data 		= LSTM_DataPreparation( Data			= StockData, 
												TimeSteps 		= 10, 
												FutureTimeSteps = pNoFuturePredictions
						)
	
	# Separate the data into training and testing
	X_train, X_test, \
	y_train, y_test 	= LSTM_sampling( x=X_data, y=y_data, TestingRecords = 5)
	
	# Run the model
	lstm_model, mDur	= LSTM_MultiModel(X_train 		  = X_train, 
										  y_train 		  = y_train, 
										  TimeSteps		  = X_train.shape[1], 
										  TotalFeatures   = X_train.shape[2], 
										  FutureTimeSteps = pNoFuturePredictions
						)
					
	# Saving the model to file
	save_path 			= './app/analysis/models'
	lstm_model.save(os.path.join(save_path, symbol_str+'_multi_lstm_model.h5'))
	
	# Update the log file re time taken to run model
	with open(os.path.join(save_path, 'log.txt'), "a") as text_file:
		text_file.write(symbol_str 							   	     + "|" + 
						datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "|" + 
						mDur									     + "\n"
		)
		
	
	return
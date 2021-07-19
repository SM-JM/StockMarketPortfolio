import os
import glob
import pandas as pd

def getFullPath( modulePath, subModulePath):
	return os.path.join( os.getcwd(),"app", modulePath, subModulePath)

def getCreatedModelSymbols():
	
	dirPath 		= getFullPath( 	modulePath	  = "analysis",
									subModulePath = "models"
					)
	
	modelFilenames 	= \
		[os.path.basename(f) for f in glob.glob(os.path.join(dirPath,"*.h5"))]
	
	return [s.split("_")[0] for s in modelFilenames]

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


def lstm_model_load (symbol):
	import 	os
	import 	glob
	import 	tensorflow 	as 		tf
	from 	tensorflow 	import 	keras
	import 	pandas		as 		pd
	import	numpy		as		np
	
	dirPath 		= getFullPath( 	modulePath	  = "analysis",
									subModulePath = "models"
					)
	
	
	##### Load models
	multi_lstm_model 	= tf.keras.models.load_model(
								os.path.join(dirPath, symbol+'_multi_lstm_model.h5')
						)
	
	#### Load dataset
	df 					= pd.read_csv(
								os.path.join('./app/analysis/historical_data',
												   'jse_main_price_2016-2020.csv')
						)
	
	
	
	#### Define DataScaler
	
	# Filter for only close prices for selected stock
	criteria			= df['Symbol'] == symbol
	data 				= df.loc[criteria,["Symbol","Close_Price"]]
	FullData			= data[["Close_Price"]].values

	# Feature Scaling for fast training of neural networks
	from sklearn.preprocessing import StandardScaler, MinMaxScaler
	 
	# Choosing between Standardization or normalization
	sc					= MinMaxScaler()
	DataScaler 			= sc.fit(FullData)

	
	
	##### Predict Price for next trading day "Jan 4, 2021"
	# Get last 10 day prices
	last10prices 		= FullData[-10:]
	# Normalise the data
	last10prices		= DataScaler.transform(last10prices.reshape(-1,1))
	
	# Changing the shape of the data to 3D
	# Choosing TimeSteps as 10 because we have used the same for training
	NumSamples=1
	TimeSteps=10
	NumFeatures=1
	last10prices		= last10prices.reshape(NumSamples,TimeSteps,NumFeatures)
	
	#############################
	# Making predictions on data
	predicted_Price 	= multi_lstm_model.predict(last10prices)
	predicted_Price 	= DataScaler.inverse_transform(predicted_Price)
	
	print(predicted_Price)
						
	return predicted_Price[-1][0] #Return the predicted price
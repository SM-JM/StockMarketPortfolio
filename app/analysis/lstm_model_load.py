import os
import glob

def getFullPath( modulePath, subModulePath):
	return os.path.join( os.getcwd(),"app", modulePath, subModulePath)

def getCreatedModelSymbols():
	
	dirPath 		= getFullPath( 	modulePath	  = "analysis",
									subModulePath = "models"
					)
	
	modelFilenames 	= \
		[os.path.basename(f) for f in glob.glob(os.path.join(dirPath,"*.h5"))]
	
	return [s.split("_")[0] for s in modelFilenames]

def lstm_model_load (dir_path):
	import 	os
	import 	glob
	import 	tensorflow 	as 		tf
	from 	tensorflow 	import 	keras
	import 	pandas		as 		pd
	import	numpy		as		np
	

	##### Load models
	multi_lstm_model 	= tf.keras.models.load_model(
								os.path.join(dir_path,'models','ccc_multi_lstm_model.h5')
						)
	
	##### Obtain summary of models
	multi_lstm_model.summary()
	
	
	#### Load dataset
	df 					= pd.read_csv(os.path.join('./app/analysis/historical_data','ccc.csv'))
	
	
	
	#### Define DataScaler
	# Extracting the closing prices of each day
	FullData			= df[['Close']].values

	# Feature Scaling for fast training of neural networks
	from sklearn.preprocessing import StandardScaler, MinMaxScaler
	 
	# Choosing between Standardization or normalization
	sc					= MinMaxScaler()
	DataScaler 			= sc.fit(FullData)

	
	
	##### Predict Price for next trading day "Jan 2, 2021"
	# Get last 10 day prices
	last10prices 		= np.array(
								df['Close'].tail(10)
						)
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
	predicted_Price 	= lstm_model.predict(last10prices)
	predicted_Price 	= DataScaler.inverse_transform(predicted_Price)
	
	print(predicted_Price)
						
	return predicted_Price[-1][-1] #Return the predicted price
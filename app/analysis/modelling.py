def obtainSinglePrediction (stock):
	import numpy	as np
	import pandas	as pd
	
	# Set random seed
	import random
	seed_num					= 1234
	random.seed(seed_num)
	#							=														=

	df_price_history			= pd.read_csv("./app/analysis/historical_data/jse_main_price_2016-2020.csv")
	df_ccc_raw					= df_price_history[df_price_history["Symbol"] == "CCC"]
	df_ccc_raw["Open_Price"]	= df_ccc_raw["Close_Price"] - df_ccc_raw["Price_Change"]

	df_ccc_raw['Date']			= pd.to_datetime(df_ccc_raw['Date'])

	df_ccc_raw.set_index( "Date", inplace=True)
											
	df_ccc						= df_ccc_raw[[	"Open_Price", 
												"Today_High",
												"Today_Low",
												"Close_Price"]]
	df_ccc["Open_Price"]		= df_ccc["Open_Price"].round(2)

	## Feature Creation
	# Input Features	
	df_ccc['H-L']				= ( df_ccc['Today_High'] - df_ccc['Today_Low']	 ).round(2)
	df_ccc['O-C']				= ( df_ccc['Close_Price'] - df_ccc['Open_Price'] ).round(2)
	df_ccc['3day MA']			= ( df_ccc['Close_Price'].shift(1).rolling(window = 3).mean() ).round(2)
	df_ccc['10day MA']			= ( df_ccc['Close_Price'].shift(1).rolling(window = 10).mean() ).round(2)
	df_ccc['30day MA']			= ( df_ccc['Close_Price'].shift(1).rolling(window = 30).mean() ).round(2)
	df_ccc['Std_dev']			= ( df_ccc['Close_Price'].rolling(5).std() ).round(2)
	# df_ccc['RSI']				= ( talib.RSI(df_ccc['Close_Price'].values, timeperiod = 9) ).round(2)
	# df_ccc['Williams %R']		= ( talib.WILLR(
													# df_ccc['Today_High'].values, 
													# df_ccc['Today_Low'].values, 
													# df_ccc['Close_Price'].values, 7)
												# ).round(2)
															
	# Output Features
	df_ccc['Price_Rise']		= np.where(df_ccc['Close_Price'].shift(-1) > df_ccc['Close_Price'], 1, 0)
	# Drop all rows storing NaN values
	df_ccc = df_ccc.dropna()
	
	return
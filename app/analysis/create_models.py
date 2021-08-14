import 	os
import 	pandas		as 		pd
import	numpy		as		np	
from .create_lstm_model_multi_variate import *				

def create_models ():
	
	# Fetching the data
	raw_df 				= pd.read_csv(	os.path.join('./app/analysis/historical_data',
													 'jse_main_price_2016-2020.csv'),
										infer_datetime_format=True, 
										sep=','
						)
	
	# Group stocks to determine how often they were traded during the year
	StockData_bySymbol 	= raw_df[['Symbol','Last_Traded']] \
							.groupby(['Symbol']) \
							.count() \
							.sort_values(by=['Last_Traded'],ascending=False)
	
	# Determine and filter for stocks that traded for at least 70% of available
	# trading days/period
	working_days_year 	= 252		# Working days for 2020 used 
	no_years			= 5 		# 2016-2020
	filter				= 0.70		# Based on scope outlined in proposal
	minDaysStockTraded	= (working_days_year * no_years) * filter
	
	# Reset the index to ensure that 'Symbol' is a column
	StockData_bySymbol 	= StockData_bySymbol.reset_index()
	
	# Create a list of stocks that meet the criteria
	criteria_forInScope	= StockData_bySymbol['Last_Traded'] >= minDaysStockTraded
	inScopeStocks 		= StockData_bySymbol.loc[criteria_forInScope,"Symbol"]
	

	# List of features to cycle through
	feature_names = ['Today_High', 
		'Today_Low', 
		'Close_Price', 
		'Volume_non_block', 
		"no_days_not_traded_since_last_traded", 
		"sentiment",
		"is_regarding_financial_report", 
		"is_sold_pur_shares"
	   ]
			   
	feature_list = [
				#["Close_Price", "sentiment"],
				#feature_names,
				["Close_Price", "Volume_non_block","Today_High","Today_Low", "sentiment"]
	]
	
	# Loop, run then save models to file
	
	sStocks = inScopeStocks[21:]
	
	for symbol in sStocks:
		s_df 	= pd.read_csv(	os.path.join('./app/analysis/historical_data/dataset',
										'prices_sentiment_'+symbol+'.csv'),
										infer_datetime_format=True, 
										sep=','
				)
		for f in feature_list:
			for e in [25]:
				create_LSTM_mv_model(StockData			= s_df,
								  symbol_str			= symbol,
								  features				= f,
								  epochs				= e
				)
		
	return
	
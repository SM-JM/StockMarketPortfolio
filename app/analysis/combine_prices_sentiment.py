


### Convert dates from string to datetime to facilite date calculations
	prices_df.Date   = pd.to_datetime(prices_df.Date,   format="%Y-%m-%d").dt.date
	sen_df.post_date = pd.to_datetime(sen_df.post_date, format="%Y-%m-%d").dt.date

### Field: Previous business day
	import datetime
	from pandas.tseries.offsets import BDay # BDay is business day

	prices_df["Previous_business_day"] 	= \ 
		prices_df.Date.apply(lambda x:(x-pd.tseries.offsets.BDay(1)))
		
	prices_df["Previous_business_day"] 	= \ 
		pd.to_datetime(prices_df.Previous_business_day, format="%Y-%m-%d").dt.date

### Function: Used for field "stock_not_traded_date_list"
	def getDatesToSumSentiments (x):
	  date_list = list()
	  for i in range(start=1,stop=x['no_days_not_traded_since_last_traded']+1):
		date_list.append(
			x['Date'] - pd.tseries.offsets.BDay(i)
		)
	  return date_list


for s in prices_df.Symbol.unique():
	
	#### Create prices df for each symbol and sort by Date field
	p_df = prices_df[prices_df.Symbol == s]
	p_df = p_df.sort_values(by='Date', ascending=True)
	
	#### Create sentiment df for each symbol and sort by Date field 
	s_df = sen_df[sen_df.instrument_code == s]
	s_df = s_df.sort_values(by='post_date', ascending=True)	
	
	### Field: Determine if traded on previous business day 
	p_df["is_traded_on_Previous_business_day"] = \
		(p_df.Date.shift() == p_df.Previous_business_day )
		
		
	### Field: Calculate the previous day last traded
	temp 		 				= p_df.Date.shift()
	temp.iloc[0] 				= p_df.Date[p_df.Date.first_valid_index()]
	p_df["Previous_trade_day"]  = temp
	
	
	### Calculate the number of days that the stock did not trade for prior 
	### to the current date
	
	# Function to determine no of business days between dates inclusive
	f = lambda x: (len(pd.bdate_range(x['Previous_trade_day'], x['Date'] )))

	p_df['no_days_not_traded_since_last_traded'] = (p_df.apply(f, axis=1))

	p_df['no_days_not_traded_since_last_traded'] =  \ 
		p_df['no_days_not_traded_since_last_traded'] - 2 # All values were off by 2 days
	
	# Set the first date as zero as there is no data for it to check against
	p_df.no_days_not_traded_since_last_traded.iloc[0] = 0 
	
	
	## Field: Determine the dates that the stock did not trade for
	p_df['stock_not_traded_date_list'] = p_df.apply(getDatesToSumSentiments, axis=1)
	
	### Convert prices dataframe to 1NF by creating a separate row for each value in
	### 'stock_not_traded_date_list' field
	
	p_df = p_df.explode('stock_not_traded_date_list')
		
	### Convert fields to string to faciliate join
	p_df.stock_not_traded_date_list = \
		p_df.stock_not_traded_date_list.dt.strftime("%Y-%m-%d")
	
	### Perform join
	c_df = pd.merge( p_df[
							["Symbol","Date","Close_Price","stock_not_traded_date_list"]
						],
						s_df[["post_date","sentiment","is_regarding_financial_report","is_sold_pur_shares"]], 
                          how='left',
                          left_on="stock_not_traded_date_list",right_on="post_date")
						  
	c_df = pd.merge( c_df,
					 s_df[["post_date","sentiment","is_regarding_financial_report","is_sold_pur_shares"]], 
					  how='left',
					  left_on="Date",right_on="post_date")

	
	c_df.to_csv("prices_sentiment_"+s+".csv")
	
from flask 			import Blueprint, render_template, request
from flask 			import current_app as app
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, DatetimeTickFormatter, HoverTool
from bokeh.models.widgets import DateRangeSlider
from bokeh.layouts import layout, column
from bokeh.models.callbacks import CustomJS
from bokeh.plotting import figure, output_file, show, save
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

import 	os
import 	glob
import 	tensorflow 	as 		tf
from 	tensorflow 	import 	keras
import 	pandas		as 		pd
import	numpy		as		np

from .lstm_model_load	import getCreatedModelSymbolsNames, lstm_model_load


# Blueprint Configuration
analysis_bp = Blueprint(
    "analysis_bp", __name__,
    template_folder='templates',
    url_prefix="/analysis"
)

@app.route('/', methods=['GET'])
@analysis_bp.route("/", methods=["GET"])
@analysis_bp.route("/priceChange", methods=["GET"])
def priceChange():
	
	pCreatedModelNames =  getCreatedModelSymbolsNames()
	
	return render_template(
		"priceChange.jinja2.html", 
		hSymbols=pCreatedModelNames,
		hReturn=""
    )
    
@analysis_bp.route("/priceChangeSubmit", methods=["GET"])
def priceChangeSubmit():
	
	# Populate dropdown 
	pCreatedModelNames 		= getCreatedModelSymbolsNames()
	
	## Get input from user
	symbol 					= request.args.get("symbol")
	
	pStartDate				= request.args.get("sDate")
	pStartDate				= datetime.strptime(pStartDate, '%Y-%m-%d')

	pEndDate				= request.args.get("eDate")
	pEndDate				= datetime.strptime(pEndDate, '%Y-%m-%d')
	
	pAllPrices				= request.args.get("isReturnAllPrices")
	
	if symbol is not None:
		## Get data to calculate price change
		df 						= getPriceData(symbol)
		
		df["Date"] 				= pd.to_datetime(df["Date"], infer_datetime_format=True)

		df.set_index(df.Date, inplace=True)

		dateList = df.Date.tolist()
		if (pStartDate not in dateList) or (pEndDate not in dateList):
			pReturn = "Error, no trades done on both dates indicated."
		else:
			## Get the startPrice and endPrice as a single value
			startPrice = (df.loc[df.Date==pStartDate,["Close_Price"]]["Close_Price"].tolist())[0]
			endPrice   = (df.loc[df.Date==pEndDate,  ["Close_Price"]]["Close_Price"].tolist())[0]
			
			# Initialise percentChange then calculate if values are valid
			percentChange = 0.0
			if startPrice > 0.0 and endPrice > 0.0:
				percentChange = (endPrice-startPrice)/startPrice
			pReturn = "{:.2%}".format(percentChange)
			createPriceChangeGraph(symbol=symbol,startDate=pStartDate,endDate=pEndDate,allPrices=pAllPrices)
	else:
		pReturn = "Error, no symbol selected or invalid symbol!"
	return render_template(
		"priceChange.jinja2.html", 
		hSymbols=pCreatedModelNames,
		hReturn=pReturn
    )	
	
@analysis_bp.route("/pricePrediction", methods=["GET"])
def pricePrediction():

	
	pCreatedModelNames =  getCreatedModelSymbolsNames()
	
	pPrediction = ""
    
	return render_template(
        "pricePrediction.jinja2.html",
		hPrediction=pPrediction, 
		hSymbols=pCreatedModelNames
    )
	
@analysis_bp.route("/pricePredictionSubmit", methods=["GET"])
def pricePredictionSubmit():
	
	symbol 					= request.args.get("symbol")
	pIsUseSentimentModels	= request.args.get("sentiment")
	
	pCreatedModelNames 		= getCreatedModelSymbolsNames()
	
	if symbol is not None:
		pPrediction 		= lstm_model_load(symbol,pIsUseSentimentModels)
		createPredictionGraph(symbol,pPrediction)
	else:
		pPrediction = ""
	return render_template(
        "pricePrediction.jinja2.html",
		hIsSentimentModel=pIsUseSentimentModels,
		hPrediction=pPrediction, 
		hSymbols=pCreatedModelNames,
		hPredictedSymbol=symbol
	)
	
@analysis_bp.route("/stockRelationship", methods=["GET"])
def stockRelationship():
	
	pCreatedModelNames 		= getCreatedModelSymbolsNames()
	
	pSubmitFlag="False"

	return render_template(
		"stockRelationship.jinja2.html",
		hSymbols=pCreatedModelNames,
		hSubmitFlag=pSubmitFlag
	)
	
@analysis_bp.route("/stockRelationshipSubmit", methods=["GET"])
def stockRelationshipSubmit():
	
	pCreatedModelNames		= getCreatedModelSymbolsNames()
	
	symbolA					= request.args.get("symbolA")
	symbolB					= request.args.get("symbolB")
	
	pSubmitFlag="False"
	
	if (symbolA is not None) and (symbolB is not None):
		createStockRelationshipGraph(symbolA,symbolB)
		pSubmitFlag = "True"
		
	return render_template(
		"stockRelationship.jinja2.html",
		hSymbols=pCreatedModelNames,
		hSubmitFlag=pSubmitFlag
	)	
	
def createStockRelationshipGraph(symbolA,symbolB):	
	# Set file to output graph to
	output_file('./app/analysis/templates/graph.html')  # Render to static HTML
	
	# Get data for graphs and scale Close_Price between 0 and 1
	dfA 			= getPriceData(symbolA)
	scalerA 		= MinMaxScaler() 
	a = dfA.Close_Price.to_numpy()
	a = a.reshape(-1, 1)
	scalerA.fit(a)
	a = scalerA.transform(a)
	dfA.Close_Price = a.reshape(-1)
	
	dfB 			= getPriceData(symbolB)
	scalerB 		= MinMaxScaler() 
	b = dfB.Close_Price.to_numpy()
	b = b.reshape(-1, 1)
	scalerB.fit(b)
	b = scalerB.transform(b)
	dfB.Close_Price = b.reshape(-1)
	
	z = dfA["Date"] # No filtering is need so assign string dates to z before conversion to datetime
		
	x 	= pd.to_datetime(dfA["Date"], infer_datetime_format=True)
	x 	= x.tolist()
	datesX   = x 
	
	x2 	= pd.to_datetime(dfB["Date"], infer_datetime_format=True)
	x2 	= x2.tolist()
	datesX2   = x2
	
	valuesY  = pd.DataFrame(dfA["Close_Price"].tolist(), columns=["A"])
	valuesY2 = pd.DataFrame(dfB["Close_Price"].tolist(), columns=["A"])

	source   = ColumnDataSource(data={'x': datesX,  'y': valuesY['A'], 'z':z}) 	
	source2  = ColumnDataSource(data={'x': datesX2, 'y': valuesY2['A'], 'z':z}) 	
		
	hover = HoverTool(
		tooltips=[('Date', '@z'), ('Price', '@y')])
		
	# Create figure
	gTitle = "Price History comparision for "+symbolA+" and "+symbolB
	p = figure(	title=gTitle,
				x_axis_label="Date", x_axis_type='datetime',
				y_axis_label="Relative Price",
				plot_width = 1500, plot_height = 400,
				#tools="pan, wheel_zoom, box_zoom, reset",
				background_fill_color="#F2F2F2")
	   
	# add a line renderer
	p.line(x='x', y='y', source=source, legend_label=symbolA, line_color="royalblue", line_width=3)
	p.circle('x', y='y', source=source, legend_label=symbolA,  color="grey", size=2)

	p.line(x='x', y='y', source=source2, legend_label=symbolB, line_color="green", line_width=3)
	p.square('x', y='y', source=source2, legend_label=symbolB,  color="DimGrey", size=2)



	p.add_tools(hover)

	#Formatting
	p.title.text_font_size			  = '20pt'
	p.axis.major_label_text_font_size = '15pt'
	p.axis.axis_label_text_font_style = 'bold italic'
	p.xaxis.axis_label_text_font_size = "16pt"
	p.yaxis.axis_label_text_font_size = "16pt"
	p.legend.location = "top_left"

	
	# show the results
	show(p)	

	return

	
def getPriceData(symbol):

	#### Load dataset
	df 	= pd.read_csv(	os.path.join('./app/analysis/historical_data/dataset',
								'prices_sentiment_'+symbol+'.csv'),
								infer_datetime_format=True, 
								sep=','
		)  	
	return df

def createPredictionGraph(symbol,prediction):
	# output to file
	output_file('./app/analysis/templates/graph.html')  # Render to static HTML
	  
	# Get data for graph
	df = getPriceData(symbol)
	
	# Create figure
	p = figure(	title="Price Prediction and Price History for "+symbol,
				x_axis_label="Date", x_axis_type='datetime',
				y_axis_label="Price",
				plot_width = 1500, plot_height = 400,
				background_fill_color="#F2F2F2")
	   
	# add a line renderer
	x 	= pd.to_datetime(df["Date"], infer_datetime_format=True)
	x 	= x.tolist()
	x   = x[-100:] # Limit the amount of dates in axis to keep graph easy to read
	x.append(datetime(2021,1,4))
	
	y1 	= df["Close_Price"].tolist()
	y1  = y1[-100:] # Limit the amount of prices in axis to keep graph easy to read
	y1.append(np.nan)
	
	y2 	= [np.nan] * (len(y1)-1) # Used to ensure that x & y axis have same len
	y2.append(prediction)
	
	z = df["Date"].tolist()
	z = z[-100:]
	z.append("2021-01-04")
	
	source   = ColumnDataSource(data={'x': x, 'y': y1, 'z':z}) 	
	source2  = ColumnDataSource(data={'x': x, 'y': y2, 'z':z}) 	

	hover = HoverTool(
						tooltips=[('Date', '@z'), ('Price', '@y')]
			)
	
	p.line(x='x', y='y', source=source, legend_label="Price History", line_color="royalblue", line_width=2)
	p.circle('x', y='y', source=source, legend_label="Price History",  color="grey", size=4)

	p.circle('x', y='y', source=source2, legend_label="Prediction",    color="goldenrod", size=12)
	p.add_tools(hover)
	
	#Formatting
	p.title.text_font_size			  = '20pt'
	p.axis.major_label_text_font_size = '15pt'
	p.axis.axis_label_text_font_style = 'bold italic'
	p.xaxis.axis_label_text_font_size = "16pt"
	p.yaxis.axis_label_text_font_size = "16pt"
	p.legend.location = "top_left"
	
	# show the results
	show(p)
	return

def createPriceChangeGraph(symbol,startDate,endDate,allPrices):
	# Set file to output graph to
	output_file('./app/analysis/templates/graph.html')  # Render to static HTML
	
	# Get data for graph
	df = getPriceData(symbol)
	
	# If allPrices not specified then filter dataframe by date range
	if allPrices != "True":
		df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True) #Convert to datetime
		df = df[(df.Date >= startDate) & (df.Date <= endDate)] #Filter
		z = df["Date"].dt.strftime('%Y-%m-%d') # Get filtered dates as string for tooltip
	else:
		z = df["Date"] # No filtering is need so assign string dates to z before conversion to datetime
		
	x 	= pd.to_datetime(df["Date"], infer_datetime_format=True)
	x 	= x.tolist()
	datesX  = x 
	valuesY = pd.DataFrame(df["Close_Price"].tolist(), columns=["A"])

	source  = ColumnDataSource(data={'x': datesX, 'y': valuesY['A'], 'z':z}) 	
		
	hover = HoverTool(
		tooltips=[('Date', '@z'), ('Price', '@y')])
		
	# Create figure
	gTitle = "Price History for "+symbol+" ("+str(startDate)[0:10]+" to "+str(endDate)[0:10]+")"
	p = figure(	title=gTitle,
				x_axis_label="Date", x_axis_type='datetime',
				y_axis_label="Price",
				plot_width = 1500, plot_height = 400,
				#tools="pan, wheel_zoom, box_zoom, reset",
				background_fill_color="#F2F2F2")
	   
	# add a line renderer
	p.line(x='x', y='y', source=source, legend_label="Price History", line_color="royalblue", line_width=2)
	p.circle('x', y='y', source=source, legend_label="Price History",  color="grey", size=4)

	p.add_tools(hover)

	#Formatting
	p.title.text_font_size			  = '20pt'
	p.axis.major_label_text_font_size = '15pt'
	p.axis.axis_label_text_font_style = 'bold italic'
	p.xaxis.axis_label_text_font_size = "16pt"
	p.yaxis.axis_label_text_font_size = "16pt"
	p.legend.location = "top_left"

	
	# show the results
	show(p)	

	return

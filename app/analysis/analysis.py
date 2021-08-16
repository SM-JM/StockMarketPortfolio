from flask 			import Blueprint, render_template, request
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, DatetimeTickFormatter, HoverTool
from bokeh.models.widgets import DateRangeSlider
from bokeh.layouts import layout, column
from bokeh.models.callbacks import CustomJS
from bokeh.plotting import figure, output_file, show, save

from datetime import datetime

from .lstm_model_load	import *
from .create_lstm_model_multi_variate import *

# Blueprint Configuration
analysis_bp = Blueprint(
    "analysis_bp", __name__,
    template_folder='templates',
    url_prefix="/analysis"
)


@analysis_bp.route("/", methods=["GET"])
@analysis_bp.route("/priceChange", methods=["GET"])
def priceChange():
	
	createPriceChangeGraph(symbol="CCC",startDate="2018-01-01",endDate="2019-01-01")
	
	return render_template(
		#"graph.html"
		"priceChange.jinja2.html", hReturn=0.00
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
	
	print("[{}] is the symbol".format(symbol))
	print("[{}] is the use sentiment option selected?".format(pIsUseSentimentModels))
	
	pCreatedModelNames 		= getCreatedModelSymbolsNames()
	
	pPrediction 		= lstm_model_load(symbol,pIsUseSentimentModels)
	
	createPredictionGraph(symbol,pPrediction)
	
	return render_template(
        "pricePrediction.jinja2.html",
		hIsSentimentModel=pIsUseSentimentModels,
		hPrediction=pPrediction, 
		hSymbols=pCreatedModelNames,
		hPredictedSymbol=symbol
	)
	
@analysis_bp.route("/stockRelationship", methods=["GET"])
def stockRelationship():
	
	
	return render_template(
		"stockRelationship.jinja2.html"
	)
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

def createPriceChangeGraph(symbol,startDate,endDate):
	# output to file
	output_file('./app/analysis/templates/graph.html')  # Render to static HTML
	
	# Get data for graph
	df = getPriceData(symbol)
	
	x 	= pd.to_datetime(df["Date"], infer_datetime_format=True)
	x 	= x.tolist()
	datesX  = x #pd.date_range(start='2016-01-01', end='2020-12-31')
	valuesY = pd.DataFrame(df["Close_Price"].tolist(), columns=["A"])
	z = df["Date"]

	# keep track of the unchanged, y-axis values
	source  = ColumnDataSource(data={'x': datesX, 'y': valuesY['A'], 'z':z}) 	
	
	hover = HoverTool(
		tooltips=[('Date', '@z'), ('Price', '@y')])
		
	# Create figure
	p = figure(	title="Price History for "+symbol,
				x_axis_label="Date", x_axis_type='datetime',
				y_axis_label="Price",
				plot_width = 1500, plot_height = 400,
				tools="pan, wheel_zoom, box_zoom, reset",
				background_fill_color="#F2F2F2")
	   
	# add a line renderer
	p.line(x='x', y='y', source=source, legend_label="Price History", line_color="royalblue", line_width=2)
	p.circle('x', y='y', source=source, legend_label="Price History",  color="grey", size=2)

	p.add_tools(hover)

	#Formatting
	p.title.text_font_size			  = '20pt'
	p.axis.major_label_text_font_size = '15pt'
	p.axis.axis_label_text_font_style = 'bold italic'
	p.xaxis.axis_label_text_font_size = "16pt"
	p.yaxis.axis_label_text_font_size = "16pt"
	
	# show the results
	show(p)	

	return

from flask 			import Blueprint, render_template, request
from bokeh.plotting import figure, output_file, show
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

    return render_template(
        "priceChange.jinja2.html",hReturn=0.00
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
	
	#pPrediction = 0
	pPrediction 		= lstm_model_load(symbol,pIsUseSentimentModels)
	
	createGraph(symbol,pPrediction)
	
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

def createGraph(symbol,prediction):
	# output to file
	output_file('./app/analysis/templates/graph.html')  # Render to static HTML
	  
	  
	  
	#### Load dataset
	df 	= pd.read_csv(	os.path.join('./app/analysis/historical_data/dataset',
								'prices_sentiment_'+symbol+'.csv'),
								infer_datetime_format=True, 
								sep=','
		)  
	# create figure
	p = figure(	title="Price Prediction and Price History for "+symbol,
				x_axis_label="Date", x_axis_type='datetime',
				y_axis_label="Price",
				plot_width = 1500, plot_height = 400,
				background_fill_color="#F2F2F2")
	   
	# add a line renderer
	x 	= pd.to_datetime(df["Date"], infer_datetime_format=True)
	x 	= x.tolist()
	x   = x[-500:] 
	x.append(datetime(2021,1,4))
	
	y1 	= df["Close_Price"].tolist()
	y1  = y1[-500:]
	y1.append(np.nan)
	
	y2 	= [np.nan] * (len(y1)-1)
	y2.append(prediction)
	
	print(y2[-5:])
	
	p.line(x,  y1, legend_label="Price History", line_color="royalblue", line_width=2)
	p.circle(x, y2, legend_label="Prediction",    color="goldenrod", size=6)
	
	p.title.text_font_size			  = '20pt'
	p.axis.major_label_text_font_size = '15pt'
	p.axis.axis_label_text_font_style = 'bold italic'
	p.xaxis.axis_label_text_font_size = "16pt"
	p.yaxis.axis_label_text_font_size = "16pt"
	
	  
	# show the results
	show(p)
	return

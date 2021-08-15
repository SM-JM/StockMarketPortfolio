from flask 			import Blueprint, render_template, request
from bokeh.plotting import figure, output_file, show



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
	isUseSentimentModels	= request.args.get("sentiment")
	
	print("[{}] is the symbol".format(symbol))
	print("[{}] is the use sentiment option selected?".format(isUseSentimentModels))
	
	pCreatedModelNames 		= getCreatedModelSymbolsNames()
	
	pPrediction = 0
	pPrediction 		= lstm_model_load(symbol,isUseSentimentModels)
	
	createGraph()
	
	return render_template(
        #"pricePrediction.jinja2.html",
		'graph.html',
		hPrediction=pPrediction, 
		hSymbols=pCreatedModelNames,
		hPredictedSymbol=symbol
	)
	
@analysis_bp.route("/stockRelationship", methods=["GET"])
def stockRelationship():
	
	
	return render_template(
		"stockRelationship.jinja2.html"
	)

def createGraph():
	# output to file
	output_file('./app/analysis/templates/graph.html')  # Render to static HTML
	  
	# create figure
	p = figure(plot_width = 400, plot_height = 400)
	   
	# add a line renderer
	p.line([1, 2, 3, 4, 5], [3, 1, 2, 6, 5], 
			line_width = 2, color = "green")
	  
	# show the results
	show(p)
	return

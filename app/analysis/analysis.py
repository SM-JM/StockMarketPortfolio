from flask 			import Blueprint, render_template, request
from .modelling		import obtainSinglePrediction
from .lstm_test		import lstm_model
from .create_models import create_models

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
	
	pPrediction = 0
    
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
	#pPrediction 		= lstm_model_load(symbol)
	
	return render_template(
        "pricePrediction.jinja2.html",
		hPrediction=pPrediction, 
		hSymbols=pCreatedModelNames,
		hPredictedSymbol=symbol
	)
	
@analysis_bp.route("/stockRelationship", methods=["GET"])
def stockRelationship():
	
	# Fetching the data
	# raw_df 				= pd.read_csv(	os.path.join('./app/analysis/historical_data',
													 # 'prices_sentiment_CCC.csv'),
										# infer_datetime_format=True, 
										# sep=','
						# )
	
	# create_LSTM_mv_model(raw_df,"CCC",5)
	
	#create_models()
	
	
	return render_template(
		"stockRelationship.jinja2.html"
	)

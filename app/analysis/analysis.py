from flask 		import Blueprint, render_template
from .modelling	import obtainSinglePrediction
from .lstm_test	import lstm_model

from .lstm_model_load	import lstm_model_load

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

	#obtainSinglePrediction("CCC")
	#lstm_model()
	pPrediction = lstm_model_load()
    
	return render_template(
        "pricePrediction.jinja2.html",hPrediction=pPrediction
    )
    
@analysis_bp.route("/stockRelationship", methods=["GET"])
def stockRelationship():

    return render_template(
        "stockRelationship.jinja2.html"
    )

    
    
    
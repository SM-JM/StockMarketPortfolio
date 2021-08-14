import glob

def getModels():

	model_filenames = [f for f in glob.glob("*.h5")]

	model_symbols = list()

	for fn in model_filenames:
		model_symbols.append(fn.split("_")[0])
		
return model_symbols
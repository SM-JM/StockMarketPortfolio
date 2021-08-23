# -*- coding: utf-8 -*-
"""DataModelling (Multivariate)-CreateModels.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IBhoUwUElkasu8uEcKFirDCJoNA2DrOW
"""

########################################################################
## Author: Shane Miller (620042891)
## Sources: 
## https://thinkingneuron.com/predicting-stock-prices-using-deep-learning-lstm-model-in-python/
## https://www.relataly.com/stock-market-prediction-using-multivariate-time-series-in-python/1815/#h-summary
########################################################################

"""## Mount Google Drive """

# Commented out IPython magic to ensure Python compatibility.
currentWorkingDir = !pwd
defaultWorkingDir = "/content"

if ( currentWorkingDir[0] == defaultWorkingDir ):
  from google.colab import drive

  drive.mount('/content/drive')
      
#   %cd "/content/drive/My Drive/Colab Notebooks/stock_portfolio"
else:
  print("Currenting running app from: ")
  !pwd

import math # Mathematical functions 
import numpy as np # Fundamental package for scientific computing with Python
import pandas as pd # Additional functions for analysing and manipulating data
import time
import os
import os.path
from datetime import date, timedelta, datetime # Date Functions
from pandas.plotting import register_matplotlib_converters # This function adds plotting functions for calender dates
import matplotlib.pyplot as plt # Important package for visualization - we use this to plot the market data
import matplotlib.dates as mdates # Formatting dates
import tensorflow as tf
from tensorflow import keras



from sklearn.metrics import mean_absolute_error, mean_squared_error # Packages for measuring model performance / errors
from tensorflow.keras.models import Sequential # Deep learning library, used for neural networks
from tensorflow.keras.layers import LSTM, Dense, Dropout # Deep learning classes for recurrent and regular densely-connected layers
from tensorflow.keras.callbacks import EarlyStopping # EarlyStopping during model training
from sklearn.preprocessing import RobustScaler, MinMaxScaler # This Scaler removes the median and scales the data according to the quantile range to normalize the price data 
#import seaborn as sns
 
# To remove the scientific notation from numpy arrays
np.set_printoptions(suppress=True)

"""# Define Functions"""

def inScopeStocks():
  ### Function to determine stocks in scope for analysis/prediction
  # Fetching the data
  raw_df              = pd.read_csv(  'jse_main_price_2016-2020.csv',
                                      infer_datetime_format=True, 
                                      sep=','
                      )
  
  # Group stocks to determine how often they were traded during the year
  StockData_bySymbol  = raw_df[['Symbol','Last_Traded']] \
                          .groupby(['Symbol']) \
                          .count() \
                          .sort_values(by=['Last_Traded'],ascending=False)
  
  # Determine and filter for stocks that traded for at least 70% of available
  # trading days/period
  working_days_year   = 252       # Working days for 2020 used 
  no_years            = 5         # 2016-2020
  filter              = 0.70      # Based on scope outlined in proposal
  minDaysStockTraded  = (working_days_year * no_years) * filter
  
  # Reset the index to ensure that 'Symbol' is a column
  StockData_bySymbol  = StockData_bySymbol.reset_index()
  
  # Create a list of stocks that meet the criteria
  criteria_forInScope = StockData_bySymbol['Last_Traded'] >= minDaysStockTraded
  inScopeStocks       = StockData_bySymbol.loc[criteria_forInScope,"Symbol"]

  return   inScopeStocks

def create_LSTM_mv_model(StockData, symbol_str, features, epochs):
    
  def LSTM_mv_scaleData(StockData):
                          
      # Replace nan with zero
      df_m = StockData.fillna(0)

      # create index with date
      df_m = df_m.set_index("Date")

      # Indexing Batches
      train_df = df_m.sort_values(by=['Date']).copy()

      # We save a copy of the dates index, before we reset it to numbers
      date_index = train_df.index

      # We reset the index, so we can convert the date-index to a number-index
      train_df = train_df.reset_index(drop=True).copy()
    
      # Create the dataset with features and filter the data to the list of FEATURES
      data = pd.DataFrame(train_df)
      data_filtered = data[features]

      # We add a prediction column and set dummy values to prepare the data for scaling
      data_filtered_ext = data_filtered.copy()
      data_filtered_ext['Prediction'] = data_filtered_ext['Close_Price']
      
      # Get the number of rows in the data
      nrows = data_filtered.shape[0]
      print(nrows)

      # Convert the data to numpy values
      np_data_unscaled = np.array(data_filtered)
      print(data_filtered.tail(5))
      print(np_data_unscaled[0])
      np_data = np.reshape(np_data_unscaled, (nrows, -1))
      print(np_data.shape)

      # Transform the data by scaling each feature to a range between 0 and 1
      scaler = MinMaxScaler()
      np_data_scaled = scaler.fit_transform(np_data_unscaled)

      # Creating a separate scaler that works on a single column for scaling predictions
      scaler_pred = MinMaxScaler()
      df_Close = pd.DataFrame(data_filtered_ext['Close_Price'])
      np_Close_scaled = scaler_pred.fit_transform(df_Close)
      
      
      return (np_data_scaled,
              data_filtered,
              data.columns.get_loc("Close_Price"),
              scaler_pred,
              date_index
      )

  np_data_scaled, \
  data_filtered,  \
  index_Close,    \
  scaler_pred,    \
  date_index      = LSTM_mv_scaleData(StockData)
    
  def LSTM_mv_DataPreparation (sequence_length):

      # sequence_length: Next few day's Price Prediction is based on the no of previous day's prices
      #            that is set here.

      # Split the training data into train and train data sets
      # As a first step, we get the number of rows to train the model on 80% of the data 
      train_data_len = math.ceil(np_data_scaled.shape[0] * 0.8)
      print(train_data_len)

      # Create the training and test data
      train_data = np_data_scaled[0:train_data_len, :]
      print(train_data[0][0])
      test_data  = np_data_scaled[train_data_len - sequence_length:, :]

      # The RNN needs data with the format of [samples, time steps, features]
      # Here, we create N samples, sequence_length time steps per sample, and the no of features
      def partition_dataset(sequence_length, data):
          x, y = [], []
          data_len = data.shape[0]
          for i in range(sequence_length, data_len):
              x.append(data[i-sequence_length:i,:]) #contains sequence_length values 0-sequence_length * columns
              y.append(data[i, index_Close]) #contains the prediction values for validation
          
          # Convert the x and y to numpy arrays
          x = np.array(x)
          y = np.array(y)
          return x, y

      # Generate training data and test data
      x_train, y_train = partition_dataset(sequence_length, train_data)
      x_test, y_test   = partition_dataset(sequence_length, test_data)    
      
      print(index_Close)

      # Print the shapes: the result is: (rows, training_sequence, features) (prediction value, )
      print(x_train.shape, y_train.shape)
      print(x_test.shape, y_test.shape)

      # Validate that the prediction value and the input match up
      # The last close price of the second input sample should equal the first prediction value
      print(x_train[1][sequence_length-1][index_Close])
      print(y_train[0])


      return ( x_train, x_test,
                y_train, y_test
      )   
    
  x_train, x_test, \
  y_train, y_test = LSTM_mv_DataPreparation(  sequence_length = 50 )

  def LSTM_mv_MultiModel( epochs ):

      # Configure the neural network model
      model = Sequential()

      # Model with n_neurons = inputshape Timestamps, each with x_train.shape[2] variables
      n_neurons = x_train.shape[1] * x_train.shape[2]
      print(n_neurons, x_train.shape[1], x_train.shape[2])
      model.add(LSTM( n_neurons, 
                      return_sequences=True, 
                      input_shape=(x_train.shape[1], x_train.shape[2])
                  )
              )    
      model.add(LSTM( n_neurons, 
                      return_sequences=False)
              )
      
      model.add(Dense(5))     
      model.add(Dense(1))

      # Compile the model
      model.compile(optimizer='adam', loss='mse')

        
      ##### Measuring the time taken by the model to train
      StartTime           = time.time()
      
      early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
      
      # Fitting the RNN to the Training set
      model.fit(      x_train, 
                      y_train, 
                      batch_size = 16, 
                      epochs = epochs,
                      validation_data=(x_test, y_test)
      )
      
      EndTime             = time.time()
      #####
      
      
      # Return the model and the total time taken to run the model
      return ( model, round((EndTime-StartTime)/60)) 


  # Run the model
  lstm_model, mDur    = LSTM_mv_MultiModel( epochs = epochs )

  def LSTM_mv_ModelEvaluation():
  
      # Get the predicted values
      y_pred_scaled = lstm_model.predict(x_test)

      # Unscale the predicted values
      y_pred = scaler_pred.inverse_transform(y_pred_scaled)
      y_test_unscaled = scaler_pred.inverse_transform(y_test.reshape(-1, 1))

      # Mean Absolute Error (MAE)
      MAE = mean_absolute_error(y_test_unscaled, y_pred)
      print(f'Median Absolute Error (MAE): {np.round(MAE, 2)}')

      # Median Absolute Error (MedAE)
      MEDAE = np.median(abs(y_pred - y_test_unscaled))
      print(f'Median Absolute Error (MedAE): {np.round(MEDAE, 2)}')

      # Mean Squared Error (MSE)
      MSE = np.square(np.subtract(y_pred, y_test_unscaled)).mean()
      print('Mean Squared Error (MSE): ' + str(np.round(MSE, 2)))

      # Root Mean Squarred Error (RMSE) 
      RMSE = np.sqrt(np.mean(np.square(y_pred - y_test_unscaled)))
      print('Root Mean Squared Error (RMSE): ' + str(np.round(RMSE, 2)))

      # Mean Absolute Percentage Error (MAPE)
      MAPE = np.mean((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled))) * 100
      print(f'Mean Absolute Percentage Error (MAPE): {np.round(MAPE, 2)} %')

      # Median Absolute Percentage Error (MDAPE)
      MDAPE = np.median((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled)) ) * 100
      print(f'Median Absolute Percentage Error (MDAPE): {np.round(MDAPE, 2)} %')
  
      return (MAE, MEDAE, MSE, RMSE, MAPE, MDAPE)
    
  MAE, MEDAE, MSE, RMSE, MAPE, MDAPE = LSTM_mv_ModelEvaluation()
    
  def createGraph(featureCount, epochs):
    # The date from which on the date is displayed
    display_start_date = pd.Timestamp(year=2021,month=1,day=1) - timedelta(days=500)

    # Add the date column
    data_filtered_sub = data_filtered.copy()

    data_filtered_sub['Date'] = date_index

    # Add the difference between the valid and predicted prices
    train_data_len = math.ceil(np_data_scaled.shape[0] * 0.8)
    train = data_filtered_sub[:train_data_len + 1]
    valid = data_filtered_sub[train_data_len:]
        
    # Get the predicted values
    y_pred_scaled = lstm_model.predict(x_test)
    y_pred = scaler_pred.inverse_transform(y_pred_scaled)
    valid.insert(1, "Prediction", y_pred.ravel(), True)
    valid.insert(1, "Difference", valid["Prediction"] - valid["Close_Price"], True)

    valid['Date'] = pd.to_datetime(valid.Date,infer_datetime_format=True)
    train['Date'] = pd.to_datetime(train.Date,infer_datetime_format=True)

    # Zoom in to a closer timeframe
    valid = valid[valid['Date'] >= display_start_date]
    train = train[train['Date'] >= display_start_date]

    # Visualize the data
    fig, ax1 = plt.subplots(figsize=(22, 10), sharex=True)
    xt = train['Date']; yt = train[["Close_Price"]]
    xv = valid['Date']; yv = valid[["Close_Price", "Prediction"]]
    plt.title("Predictions vs Actual Values", fontsize=20)
    plt.ylabel(symbol_str, fontsize=18)
    plt.plot(xt, yt, color="#039dfc", linewidth=2.0)
    plt.plot(xv, yv["Prediction"], color="#E91D9E", linewidth=2.0)
    plt.plot(xv, yv["Close_Price"], color="black", linewidth=2.0)
    plt.legend(["Train", "Test Predictions", "Actual Values"], loc="upper left")

    # # Create the bar plot with the differences
    x = valid['Date']
    y = valid["Difference"]

    # Create custom color range for positive and negative differences
    valid.loc[y >= 0, 'diff_color'] = "#2BC97A"
    valid.loc[y < 0, 'diff_color'] = "#C92B2B"

    plt.bar(x, y, width=0.8, color=valid['diff_color'])
    plt.grid()

    fn = '_'.join(["mv", symbol_str, str(featureCount), str(epochs)+".png"])

    plt.savefig( "./models/"+fn, bbox_inches='tight')

    fig.clear()
    plt.close(fig)

    return
  
  createGraph(len(features), epochs)
  eval_list = [str(np.round(MAE,4)),
              str(np.round(MEDAE,4)),
              str(np.round(MSE,4)),
              str(np.round(RMSE,4)),
              str(np.round(MAPE,4)),
              str(np.round(MDAPE,4))]
  
  save_path           = './models'
  log_path            = os.path.join(save_path, 'log.csv')
    
  def writeModeltoFile():
  
      # Create a string of features
      features_str = '|'.join([str(c) for c in data_filtered.columns.tolist() ])
      features_count = str(np_data_scaled.shape[1])
      
      fn ="mv_" + symbol_str + "_" + str(features_count) + "_" + str(epochs) + '_lstm_model.h5'

      # Saving the model to file
      lstm_model.save(os.path.join(save_path, fn))
      

      # Update the log file re time taken to run model
      with open(log_path, "a") as text_file:
          text_file.write(fn                                           + "," +
                          symbol_str                                   + "," + 
                          datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "," + 
                          ','.join(eval_list)                          + "," +
                          features_str                                 + "," +
                          features_count                               + "," +
                          str(epochs)                                  + "," +
                          str(mDur)                                    + "\n"
          )   
  
  if os.path.isfile(log_path):
      writeModeltoFile()
  else:
      with open(log_path, "a") as text_file:
          text_file.write(','.join(["Filename",
                                  "Symbol",
                                  "Date_Model_Completed",
                                  "MAE","MEDAE","MSE","RMSE","MAPE","MDAPE",
                                  "Feature_Names",
                                  "Feature_Count",
                                  "epochs",
                                  "Training_Duration_minutes\n"]))
      writeModeltoFile()
  
  return

"""# Create models then save the model, its graph to file, and log it in log.csv

## Search for best Hyperparameter for LSTM model
"""

feature_list = [
        ["Close_Price", "sentiment"],
        ["Today_High","Today_Low","Close_Price","Volume_non_block","no_days_not_traded_since_last_traded","sentiment","is_regarding_financial_report","is_sold_pur_shares"],
        ["Today_High","Today_Low","Close_Price", "Volume_non_block"]
  ]

sStocks = inScopeStocks()
print(sStocks)

for i in [0]:

    
  # Loop, run then save models to file
  # Only three stocks are used based on the time it takes to run each model
  for symbol in sStocks[0:4]:
    s_df 	= pd.read_csv(	os.path.join('./',
                    'prices_sentiment_'+symbol+'.csv'),
                    infer_datetime_format=True, 
                    sep=','
        )
    for f in feature_list:
      for e in [25, 50]:
        create_LSTM_mv_model(
                  StockData			= s_df,
                  symbol_str		= symbol,
                  features			= f,
                  epochs				= e
        )

"""## Based on analysis of results above, model with epochs of 25 and the features below are the best models for us to use for our analysis"""

feature_list = [
        ["Today_High","Today_Low","Close_Price", "Volume_non_block"],
        ["Today_High","Today_Low","Close_Price", "Volume_non_block","sentiment"]
]

sStocks = inScopeStocks()
print(sStocks)

# Loop, run then save models to file
for symbol in sStocks:
  s_df 	= pd.read_csv(	os.path.join('./',
                  'prices_sentiment_'+symbol+'.csv'),
                  infer_datetime_format=True, 
                  sep=','
      )
  for f in feature_list:
    for e in [25]:
      create_LSTM_mv_model(
                StockData			= s_df,
                symbol_str		= symbol,
                features			= f,
                epochs				= e
      )

"""# Save System Specs"""

!df -h > "./models/filesystem.txt"

!cat /proc/cpuinfo > "./models/cpuinfo.txt"

!cat /proc/meminfo > "./models/meminfo.txt"

cat /etc/os-release > "./models/os_description.txt"

!uname -a > "./models/os_bit.txt"
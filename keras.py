import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam

from api import API

import talib
import random

"""
TODO: fix output
"""

def showFrame(data):
    print(data.describe())
    data.plot()
    plt.show()

def showArray(title, label,data):
    fig1 = plt.figure(label)
    ax = fig1.gca()
    plt.ylabel(label)
    plt.title(title)
    ax.plot(data)

def plotStockData(title, data):
    showArray(title, 'open', data[:, 0])
    showArray(title, 'high', data[:, 1])
    showArray(title, 'low', data[:, 2])
    showArray(title, 'close', data[:, 3])
    plt.show()

def getDataSet():
	dataset = pd.read_json(API.getJsonFromFile('aapl_data.json'), 'records')
	#dataset = dataset.drop(['changeOverTime', 'date', 'high', 'low', 'marketClose', 'average',
	#			   'marketNotional', 'marketOpen', 'notional', 'numberOfTrades', 'open', 'volume', 'close', 'minute', 'label',
	#						  'marketHigh', 'marketLow'], 1)
	dataset = dataset.dropna()
	dataset = dataset[['open', 'high', 'low', 'close']]
	#dataset = dataset.values
	#plotStockData('AAPL',dataset)
	dataset['H-L'] = dataset['high'] - dataset['low']
	dataset['O-C'] = dataset['close'] - dataset['open']
	dataset['3day MA'] = dataset['close'].shift(1).rolling(window = 3).mean()
	dataset['10day MA'] = dataset['close'].shift(1).rolling(window = 10).mean()
	dataset['30day MA'] = dataset['close'].shift(1).rolling(window = 30).mean()
	dataset['Std_dev']= dataset['close'].rolling(5).std()
	dataset['RSI'] = talib.RSI(dataset['close'].values, timeperiod = 9)
	dataset['Williams %R'] = talib.WILLR(dataset['high'].values, dataset['low'].values, dataset['close'].values, 7)
	dataset['Price_Rise'] = np.where(dataset['close'].shift(-1) > dataset['close'], 1, 0)
	dataset = dataset.dropna()
	return dataset
	
def main():
	#data = API.getMonthsData("AAPL")
	#API.printToFileJson("aapl_data.json", data)
	#return
	dataset = getDataSet()
	
	X = dataset.iloc[:, 4:-1]
	y = dataset.iloc[:, -1]

	split = int(len(dataset)*0.8)
	X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
	
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)
	print(X_train)
	print(X_test)
	#return
	classifier = Sequential()
	classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu', input_dim = X.shape[1]))
	classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu'))
	classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
	customizedOptimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	#customizedOptimizer = SGD(lr=0.000000000000001, momentum=0.0, decay=0.0, nesterov=False)
	classifier.compile(optimizer = customizedOptimizer, loss = 'mean_squared_error', metrics = ['accuracy'])

	classifier.fit(X_train, y_train, batch_size = 15, epochs = 150)

	
	y_pred = classifier.predict(X_test)
	y_pred = (y_pred > 0.5)

	dataset['y_pred'] = np.NaN
	dataset.iloc[(len(dataset) - len(y_pred)):,-1:] = y_pred
	trade_dataset = dataset.dropna()

	
	trade_dataset['actual'] = 0.
	trade_dataset['actual'] = np.log(trade_dataset['close']/trade_dataset['close'].shift(1))
	trade_dataset['actual'] = trade_dataset['actual'].shift(-1)

	trade_dataset['predicted'] = 0.
	trade_dataset['predicted'] = np.where( trade_dataset['y_pred'] == True, trade_dataset['actual'], - trade_dataset['actual'])

	trade_dataset['Cumulative Actual Values'] = np.cumsum(trade_dataset['actual'])
	trade_dataset['Cumulative Predicted Values'] = np.cumsum(trade_dataset['predicted'])

	plt.figure(figsize=(10,5))
	plt.plot(trade_dataset['Cumulative Actual Values'], color='r', label='Actual Values')
	plt.plot(trade_dataset['Cumulative Predicted Values'], color='g', label='Predicted Values')
	plt.legend()
	plt.show()
	
main()
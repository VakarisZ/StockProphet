import urllib.request
import json
from datetime import timedelta, date
import numpy as np
import pandas

class API(object):

    @staticmethod
    def getStockData(symbol, range = "1d", date = None ):
        """
        :param symbol: stock symbol. Ex: APPL
        :param range: price range. 1d, 1m, 3m, 6m [months]...
        :param date: days prices each minute. Format: YYYYMMDD
        :return: list of dicts from API
        """
        if date != None:
            link = "https://api.iextrading.com/1.0/stock/%s/chart/date/%s" % (symbol, date)
        else:
            link = "https://api.iextrading.com/1.0/stock/%s/chart/%s" % (symbol, range)
        response = json.loads(urllib.request.urlopen(link).read())
        return response

    @staticmethod
    def getMonthsData(symbol):
        """
        Gets every minute of market data for as long as API has records for(1 month)
        :param symbol: stock symbol. Ex: APPL
        :return: months worth list of minutes in stock market
        """
        missed = 0
        iter_date = date.today()
        data = []
        while missed < 5:
            resp = API.getStockData(symbol, "1d", iter_date.strftime("%Y%m%d"))
            if not resp:
                missed += 1
            else:
                missed = 0
                for idx,min in enumerate(resp):
                    if min['marketAverage'] == 0 or min['marketAverage'] == -1:
                        temp = resp[idx-1]
                        temp['marketVolume'] = 0
                        temp['marketNumberOfTrades'] = 0
                        resp[idx] = temp
                data = data + resp
            iter_date -= timedelta(1)
        return data

    @staticmethod
    def printToFileVerbose(filename, data):
        results = open(filename, 'w')
        for minute in data:
            results.write("Date: %s; Minute: %s; High: %s; Low: %s;"
                          "TradesNum: %s; Volume: %s\n" % (minute['date'], minute['minute'],
                                                           minute['marketHigh'], minute['marketLow'],
                                                           minute['marketNumberOfTrades'],
                                                           minute['marketVolume']))

    @staticmethod
    def printToFileJson(filename, data):
        results = open(filename, 'w')
        results.write(json.dumps(data))

    @staticmethod
    def getJsonFromFile(filename):
        return open(filename, 'r').read()

    @staticmethod
    def prepareData(data, mins_in, mins_out):
        """
        :param data: x(features)*y(minutes length) data set
        :param mins_in: how many minutes with all features in a row in output (min*features)
        :param mins_out: how many minutes to predict
        :return: X - 2d array x(minute after minute), y(input count)
                 Y - 2d array x(averages of n pretictable minutes), y(input count)
        """
        x = len(data[0])
        y = len(data)
        flat_data = np.ravel(data)
        cols_in = mins_in * x
        rows_in = (len(data) - mins_in - mins_out)
        rows_out = rows_in
        cols_out = mins_out * x
        training = [[0 for x in range(cols_in)] for y in range(rows_in)]
        targets = [[0 for x in range(mins_out)] for y in range(rows_in)]
        for idx, val in enumerate(training):
            start = idx * 3
            if idx < y:
                training[idx][:] = flat_data[start:start + cols_in]
                temp = flat_data[start + cols_in:start + cols_in + cols_out]
                targets[idx][:] = temp[::x]
            else:
                break
        return [training, targets]

    @staticmethod
    def getPreparedData(mins_in, mins_out):
        all_data = pandas.read_json(API.getJsonFromFile('atvi_new.json'), 'records')
        all_data = all_data.drop(
            ['changeOverTime', 'date', 'high', 'low', 'marketChangeOverTime', 'marketClose', 'average',
             'marketNotional', 'marketOpen', 'notional', 'numberOfTrades', 'open', 'volume', 'close', 'minute', 'label',
             'marketHigh', 'marketLow'], 1)
        all_data = all_data.values
        all_data = API.prepareData(all_data, mins_in, mins_out)
        return all_data

import urllib.request
import json
from datetime import timedelta, date


class API(object):
    def __init__(self):
        pass
    def getStockData(self, symbol, range = "1d", date = None ):
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

    def getMonthsData(self, symbol):
        """
        Gets every minute of market data for as long as API has records for(1 month)
        :param symbol: stock symbol. Ex: APPL
        :return: months worth list of minutes in stock market
        """
        missed = 0
        iter_date = date.today()
        data = []
        while missed < 5:
            resp = self.getStockData(symbol, "1d", iter_date.strftime("%Y%m%d"))
            if not resp:
                missed += 1
            else:
                missed = 0
                data = data + resp
            iter_date -= timedelta(1)
        return data
    def printToFileVerbose(self):
        results = open('ATVI_data', 'w')
        for minute in data:
            results.write("Date: %s; Minute: %s; High: %s; Low: %s;"
                          "TradesNum: %s; Volume: %s\n" % (minute['date'], minute['minute'],
                                                           minute['marketHigh'], minute['marketLow'],
                                                           minute['marketNumberOfTrades'],
                                                           minute['marketVolume']))
    def printToFileJson(self, filename, data):
        results = open(filename+".json", 'w')
        results.write(json.dumps(data))


api = API()
data = api.getMonthsData("ATVI")
api.printToFileJson("atvi_data", data)


import urllib.request
import json

class API(object):
    def __init__(self):
        pass
    def getStockData(self, symbol, range = "1d", date = None ):
        """
        :param symbol: stock symbol. Ex: APPL
        :param range: price range. 1d, 1m, 3m, 6m [months]...
        :param date: days prices each minute
        :return: list of dicts from API
        """
        if date != None:
            link = "https://api.iextrading.com/1.0/stock/%s/chart/date/%s" % (symbol, date)
        else:
            link = "https://api.iextrading.com/1.0/stock/%s/chart/%s" % (symbol, range)
        response = json.loads(urllib.request.urlopen(link).read())
        return response
api = API()
data = api.getStockData("ATVI", "1m")


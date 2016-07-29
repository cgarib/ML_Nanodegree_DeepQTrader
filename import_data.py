import urllib
import pandas as pd
import numpy as np

def fetch_data(symbol):

    time_frame = "d" # d -> daily, w -> weekly, m -> monthly.
    url = "http://real-chart.finance.yahoo.com/table.csv?s="+symbol+\
            "&a=11&b=22&c=1998&d=04&e=9&f=2016&g="+time_frame+"+&ignore=.csv"

    urllib.urlretrieve(url, './raw_data/{}.csv'.format(symbol))
    print "Downloading for "+symbol

def DownloadPrices():
    list = pd.read_csv("list.csv")
    for symbol in list["SYMBOL"]:
     fetch_data(symbol)

if __name__ == "__main__":
    #list=pandas.read_csv("list.csv")
    #for symbol in list["SYMBOL"]:
        #fetch_data(symbol)
    prices = pd.read_pickle("data/prices.pkl")
    print prices.head()

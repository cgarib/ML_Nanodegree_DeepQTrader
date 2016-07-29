import os
import pandas as pd
import numpy as np
import import_data

def fill_missing_values(df_data):
    df_data.fillna(method="ffill",inplace="TRUE")
    df_data.fillna(method="bfill", inplace="TRUE")


def symbol_to_path(symbol, base_dir="raw_data"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        print "Fetching {}".format(symbol)
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df


def get_rolling_mean(values, window,min_periods=None):
    return pd.rolling_mean(values, window=window, min_periods=min_periods)


def get_rolling_std(values, window,min_periods=None):
    """Return rolling standard deviation of given values, using specified window size."""
    return pd.rolling_std(values, window=window, min_periods=min_periods)


def get_bollinger_bands(rm, window,min_periods=None):
    rstd=get_rolling_std(rm, window, min_periods)
    upper_band=rm+rstd*2
    lower_band = rm - rstd * 2
    return upper_band, lower_band

def getAllStocks():
    list = pd.read_csv("list.csv")
    dates = pd.date_range('2010-01-01', '2016-05-09')
    all = get_data(list["SYMBOL"].values.tolist(),dates)
    print(all.head())
    all.to_pickle('data/all_unprocessed.pkl')
    print("Saved!")

def getPrices():
    list = pd.read_csv("list.csv")
    dates = pd.date_range('2010-01-01', '2016-05-09')
    all = get_data(list["SYMBOL"].values.tolist(),dates)
    fill_missing_values(all)
    all.to_pickle('data/prices.pkl')
    print("{} Prices Saved!".format(len(all)))
    return all

def CalculateAdjSMAs(df):
    for w in [10, 20, 50, 200]:
        sma = get_rolling_mean(df, w, 0)
        adjsma = df / sma-1
        adjsma.to_pickle('data/adjsma{}.pkl'.format(w))
        print("AdjSMA {} Saved!".format(w))

def CalculateBollingerBands(df):
    for w in [10, 20, 50, 200]:
        bbu,bbl = get_bollinger_bands(df, w,0)
        bbu.fillna(method="bfill", inplace="TRUE")
        bbl.fillna(method="bfill", inplace="TRUE")
        bbub=bbu<df
        bblb = bbl > df
        bbub=bbub.applymap(lambda x: 1 if x else 0)
        bblb=bblb.applymap(lambda x: 1 if x else 0)
        bb=bbub-bblb
        bb.to_pickle('data/bb{}.pkl'.format(w))

        print("Bollinger Bands {} Saved!".format(w))

def CalculateBollingerBands(df):
    for w in [10, 20, 50, 200]:
        bbu,bbl = get_bollinger_bands(df, w,0)
        bbu.fillna(method="bfill", inplace="TRUE")
        bbl.fillna(method="bfill", inplace="TRUE")
        bbub=bbu<df
        bblb = bbl > df
        bbub=bbub.applymap(lambda x: 1.0 if x else 0.0)
        bblb=bblb.applymap(lambda x: 1.0 if x else 0.0)
        bb=bbub-bblb
        bb.to_pickle('data/bb{}.pkl'.format(w))

        print("Bollinger Bands {} Saved!".format(w))


def CalculateStates(df,state_size_day,datanames,date_test,save=True):
    date_test_conv=np.datetime64(date_test+'T21:00:00.000000000-0300')
    data = []
    train_states = pd.DataFrame(columns=["State"])
    test_states = pd.DataFrame(columns=["State"])
    nfeatures=0

    print("Calculating States!")

    for name in datanames:
        data.append(pd.read_pickle('data/{}.pkl'.format(name)))

    for i in range(state_size_day-1,len(df)):
        rest_of_features=[]
        for d in data:
            for j in range(i-state_size_day+1,i+1):
                rest_of_features.extend(d.iloc[j].values)
        state=[df.index.values[i]]
        state.extend(rest_of_features)
        nfeatures = len(state)
        if state[0]<=date_test_conv:
            train_states.loc[len(train_states)] = [state]
        else:
            test_states.loc[len(test_states)] = [state]
    if save:
        print "Saving Train Data"
        train_states.to_pickle("data/train.pkl")
        print "Saving Test Data"
        test_states.to_pickle("data/test.pkl")
        print "States Saved! {} dimensions".format(nfeatures)
    return train_states,test_states







if __name__ == "__main__":
    import_data.DownloadPrices()
    df=getPrices()
    CalculateAdjSMAs(df)
    CalculateBollingerBands(df)
    CalculateStates(df,1,['adjsma20','bb20'],'2016-01-01')
    #getAllData(7,['adjsma20','bb20'])
    #print(getRandomHoldCombinations(10))


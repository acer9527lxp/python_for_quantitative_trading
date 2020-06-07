# _*_ coding:utf-8 _*_
# @Time     : 2020/5/30 13:17
# @Auhtor   : laixinping
# @Emial    : xinping_lai@126.com
# @File     : datahandler.py
# @software : PyCharm

import time
import tushare as ts
import os, os.path
import pandas as pd
from event import *
from abc import ABCMeta, abstractmethod


# data.py

class DataHandler(object):
    """
    DataHandler is an abstract base class providing an interface for
    all subsequent (inherited) data handlers (both live and historic).

    The goal of a (derived) DataHandler object is to output a generated
    set of bars (OLHCVI) for each symbol requested.

    This will replicate how a live strategy would function as current
    market data would be sent "down the pipe". Thus a historic and live
    system will be treated identically by the rest of the backtesting suite.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_latest_bars(self, symbol, N=1):
        """
        Returns the last N bars from the latest_symbol list,
        or fewer if less bars are available.
        """
        raise NotImplementedError("Should implement get_latest_bars()")

    @abstractmethod
    def update_bars(self):
        """
        Pushes the latest bar to the latest symbol structure
        for all symbols in the symbol list.
        """
        raise NotImplementedError("Should implement update_bars()")


# data.py

class HistoricCSVDataHandler(DataHandler):
    """
    HistoricCSVDataHandler is designed to read CSV files for
    each requested symbol from disk and provide an interface
    to obtain the "latest" bar in a manner identical to a live
    trading interface.
    """

    def __init__(self, events, csv_dir, symbol_list):
        """
        Initialises the historic data handler by requesting
        the location of the CSV files and a list of symbols.

        It will be assumed that all files are of the form
        'symbol.csv', where symbol is a string in the list.

        Parameters:
        events - The Event Queue.
        csv_dir - Absolute directory path to the CSV files.
        symbol_list - A list of symbol strings.
        """
        self.events = events # 时间队列
        self.csv_dir = csv_dir # 数据所在目录
        self.symbol_list = symbol_list # 股票列表

        self.symbol_data = {} #数据集合 {股票1: df1,股票2: df2,....}
        self.latest_symbol_data = {} # 是self.symbol_data中每个股票的最新N条数据 形式和self.symbol_data一样
        self.continue_backtest = True

        self._open_convert_csv_files()

    def _open_convert_csv_files(self):
        """
        Opens the CSV files from the data directory, converting
        them into pandas DataFrames within a symbol dictionary.
        用字典的形式加载所有股票的数据，所有数据的表头是一致的

        For this handler it will be assumed that the data is
        taken from DTN IQFeed. Thus its format will be respected.
        """
        comb_index = None
        for s in self.symbol_list:
            # Load the CSV file with no header information, indexed on date
            self.symbol_data[s] = pd.io.parsers.read_csv(
                                      os.path.join(self.csv_dir, '%s.csv' % s),
                                      header=0, index_col=0,
                                      names=['datetime','open','close','high','low','volume']
                                  )

            # Combine the index to pad forward values 因为多只股票的交易日期很可能是不一致的，所有为了处理缺失日期的数据统一日期索引
            if comb_index is None:
                comb_index = self.symbol_data[s].index
            else:
                comb_index.union(self.symbol_data[s].index) # 所有的 symbol 的 index 都拼接起来 组成一份最全的时间序列

            # Set the latest symbol_data to None
            self.latest_symbol_data[s] = []

        # Reindex the dataframes 根据最全的时间序列，填充每一个symbol的数据框，保持数据的一致性 method='pad' 用前一条填充下一条缺失数据
        for s in self.symbol_list:
            self.symbol_data[s] = self.symbol_data[s].reindex(index=comb_index, method='pad').iterrows()



    def _get_new_bar(self, symbol):
        """ 根据指定的symbol 返回该股票的数据. 是一个生成迭代器, 每次都返回一条数据.
        Returns the latest bar from the data feed as a tuple of
        (sybmbol, datetime, open, low, high, close, volume). bar:指的是一条数据记录。
                      比如是一天的数据或者是一小时又或是一分钟，表示一个市场数据
                      ? self.symbol_data[symbol] 是一个迭代器 for循环不是返回了所有数据吗？
        """
        for b in self.symbol_data[symbol]:
            yield tuple([symbol, pd.to_datetime(b[0]),
#                          datetime.datetime.strptime(b[0], '%Y-%m-%d %H:%M:%S'),
                        b[1][0], b[1][1], b[1][2], b[1][3], b[1][4]])



    def get_latest_bars(self, symbol, N=1):
        """
        Returns the last N bars from the latest_symbol list,
        or N-k if less available.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
        else:
            return bars_list[-N:]


    def update_bars(self):
        """
        Pushes the latest bar to the latest_symbol_data structure
        for all symbols in the symbol list.
        """
        for s in self.symbol_list:
            try:
                bar = self._get_new_bar(s).__next__()
            except StopIteration:
                self.continue_backtest = False
            else:
                if bar is not None:
                    self.latest_symbol_data[s].append(bar)
        self.events.put(MarketEvent())


if __name__ == "__main__":
    SavePath = "../StockData/"
    cols = ['datetime', 'open', 'close', 'high', 'low', 'volume']

    for symbol in ['000063', "002203", "601318"]:
        df = ts.get_k_data(symbol)

        df.rename(columns={"date": "datetime"}, inplace=True)
        df.drop("code", axis=1, inplace=True)

        filename = f"{symbol}.csv"
        df.to_csv(os.path.join(SavePath, filename), index=False, encoding='utf8')



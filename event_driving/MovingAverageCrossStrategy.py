# _*_ coding:utf-8 _*_
# @Time     : 2020/5/30 13:28
# @Auhtor   : laixinping
# @Emial    : xinping_lai@126.com
# @File     : MovingAverageCrossStrategy.py
# @software : PyCharm

import datetime
import numpy as np
import pandas as pd
from backtest import Backtest
from datahandler import HistoricCSVDataHandler
from event import SignalEvent
from execution import SimulatedExecutionHnadler
from portfolio import Portfolio, NaiveProtfolio
from strategy import Strategy


class MovingAverageCrossStrategy(Strategy):
    """
    Carries out a basic Moving Average Crossover strategy with a
    short/long simple weighted moving average. Default short/long
    windows are 100/400 periods respectively.
    """

    def __init__(self, bars, events, short_window=30, long_window=90):
        """
        Initialises the buy and hold strategy.

        Parameters:
        bars - The DataHandler object that provides bar information
        events - The Event Queue object.
        short_window - The short moving average lookback.
        long_window - The long moving average lookback.
        """
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events
        self.short_window = short_window
        self.long_window = long_window

        # Set to True if a symbol is in the market
        self.bought = self._calculate_initial_bought()

    def _calculate_initial_bought(self):
        """
        Adds keys to the bought dictionary for all symbols
        and sets them to 'OUT'.
        """
        bought = {}
        for s in self.symbol_list:
            bought[s] = 'OUT'
        return bought

    def calculate_signals(self, event):
        """
        Generates a new set of signals based on the MAC
        SMA with the short window crossing the long window
        meaning a long entry and vice versa for a short entry.

        Parameters
        event - A MarketEvent object.
        """
        if event.type == 'MARKET':
            for symbol in self.symbol_list:
                #                 bars_temp = self.bars.get_latest_bars_values(symbol, "close", N=self.long_window)
                bars = self.bars.get_latest_bars(symbol, N=self.long_window)

                #                 print(f"$$$$bars=:{bars}$$$$")

                if bars is not None and bars != []:

                    # bars 是一个list 转成dataFrame
                    latest_n_bars = pd.DataFrame(np.array(bars),
                                                 columns=['code', 'datetime', 'open', 'close', 'high', 'low', 'volume'])
                    latest_n_bars['datetime'] = pd.to_datetime(latest_n_bars['datetime'])
                    latest_n_bars.set_index('datetime', inplace=True)
                    bars = latest_n_bars['close']

                    #                     print(f"$$$$: {bars}$$$$")

                    short_sma = np.mean(bars[-self.short_window:])
                    long_sma = np.mean(bars[-self.long_window:])

                    dt = bars.index[-1]

                    #                     dt = self.bars.get_latest_bar_datetime(symbol)
                    sig_dir = ""
                    strength = 1.0
                    strategy_id = 1
                    # 短期均线上穿长期均线：金叉 买入
                    if short_sma > long_sma and self.bought[symbol] == "OUT":
                        sig_dir = 'LONG'
                        signal = SignalEvent(strategy_id, symbol, dt, sig_dir, strength)
                        self.events.put(signal)
                        self.bought[symbol] = 'LONG'
                    # 短期均线下穿长期均线：死叉 卖入
                    elif short_sma < long_sma and self.bought[symbol] == "LONG":
                        sig_dir = 'EXIT'
                        signal = SignalEvent(strategy_id, symbol, dt, sig_dir, strength)
                        self.events.put(signal)
                        self.bought[symbol] = 'OUT'


if __name__ == "__main__":
    csv_dir = "../StockData/"
    symbol_list = ['002203', "601318"]
    # symbol_list = ['000063', "002203", "601318"]
    initial_capital = 100000.0
    start_date = datetime.datetime(2017, 11, 1, 0, 0, 0)
    short_window = 30
    long_window = 90
    heartbeat = 0.0

    backtest = Backtest(csv_dir,
                        symbol_list,
                        initial_capital,
                        heartbeat,
                        start_date,
                        HistoricCSVDataHandler,
                        SimulatedExecutionHnadler,
                        NaiveProtfolio,
                        MovingAverageCrossStrategy)

    backtest.simulate_trading()

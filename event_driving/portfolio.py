# _*_ coding:utf-8 _*_
# @Time     : 2020/5/30 13:25
# @Auhtor   : laixinping
# @Emial    : xinping_lai@126.com
# @File     : portfolio.py
# @software : PyCharm

import datetime
import numpy as np
import pandas as pd
import queue
import os
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
from event import SignalEvent, FillEvent, OrderEvent
from math import floor
from performance import create_sharpe_ratio, create_drawdowns

class Portfolio(object):
    """
    the portfolio class handles the positions and market
    value of all instruments at a resolution of a "bar"
    i.e. secondly minutely 5-min 30-min 6--min or EOD
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def update_signal(self, event):
        """ acts on a signalement to generate new orders
        based on the portfolio logic
        """
        raise NotImplementedError("Should implement update_signal()")

    @abstractmethod
    def update_fill(self, event):
        """updates the portfolia current positions and
        holdings from a FILLEvent
        """
        raise NotImplementedError("Should implement update_fill()")


"""
The principal subject of this article is the NaivePortfolio class. 
It is designed to handle position sizing and current holdings, but will 
carry out trading orders in a "dumb" manner by simply sending them directly
 to the brokerage with a predetermined fixed quantity size, irrespective of 
 cash held. These are all unrealistic assumptions, but they help to outline 
 how a portfolio order management system (OMS) functions in an event-driven fashion.
"""


class NaiveProtfolio(Portfolio):
    """ The NaivePortfolio object is designed to send orders to
    a brokerage object with a constant quantity size blindly,
    i.e. without any risk management or position sizing. It is
    used to test simpler strategies such as BuyAndHoldStrategy.
    """

    def __init__(self, bars, evetns, start_date, initial_capital=100000.0):
        """Initialises the portfolio with bars and an event queue.
        Also includes a starting datetime index and initial capital
        (USD unless otherwise stated).

        :param bars: he DataHandler object with current market data.
        :param evetns: The Event Queue object.
        :param start_date: The start date (bar) of the portfolio.
        :param initial_capital: The starting capital in USD.
        """
        self.bars = bars
        self.events = evetns
        self.symbol_list = self.bars.symbol_list
        self.start_date = start_date
        self.initial_capital = initial_capital

        self.all_positions = self.construct_all_positions()
        self.current_positions = dict((k, v) for k, v in [(s, 0) for s in self.symbol_list])
        # {symbol1:0,symbol2:0,....}初始化持仓情况 0 当前最新的持仓情况

        self.all_holdings = self.construct_all_holdings()
        self.current_holdings = self.construct_current_holdings()

        self.savepath = "../output/"

    def construct_all_positions(self):
        """ constructs the positions list using the
        start_date to determine when the time index will begin.
        初始化持仓情况 0

        :return:[{symbol1:0,datetime:20200508},{symbol2:0,datetime:20200508},...]
        """
        d = dict((k, v) for k, v in [(s, 0) for s in self.symbol_list])
        d['datetime'] = self.start_date
        return [d]

    """
    The construct_all_holdings method is similar to the above but adds extra keys for cash, 
    commission and total, which respectively represent the spare cash in the account after any
     purchases, the cumulative commission accrued and the total account equity including cash 
     and any open positions. Short positions are treated as negative. The starting cash and total 
     account equity are both set to the initial capital
    """

    def construct_all_holdings(self):
        """Constructs the holdings list using the start_date
        to determine when the time index will begin.
        :return:[{symbol1:0.0,datetime:20200508,cash:initial_capital,commission:0.0,total:initial_capital},
                {symbol2:0.0,datetime:20200508,cash:initial_capital,commission:0.0,total:initial_capital},
             ]
            注意这里返回的是一个list，其元素是字典类型，也就是说它会记录每一次操作后的结果
        """
        d = dict((k, v) for k, v in [(s, 0.0) for s in self.symbol_list])
        d['datetime'] = self.start_date
        d['cash'] = self.initial_capital  # 可用资金
        d['commission'] = 0.0  # 累计佣金
        d['total'] = self.initial_capital
        return [d]

    def construct_current_holdings(self):
        """This constructs the dictionary which will hold the instantaneous
        value of the portfolio across all symbols.
        :return:
        """
        d = dict((k, v) for k, v in [(s, 0.0) for s in self.symbol_list])
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return d

    def update_timeindex(self, event):
        """ adds a new record to the positions maxtrix for the current
        market data bar. this reflects the PREVIOUS bar,i.e all\
        current market data at this stage is know
        makes use of a marketaEvent from the events queue
        :param event:根据最新的市场数据在 all_holdings 中增加最新的持仓记录
        :return:
        """
        bars = {}
        for sym in self.symbol_list:
            bars[sym] = self.bars.get_latest_bars(sym, N=1)  # 获取每一个股票的最新一条数据
        #             print(f"1######{bars[sym]}#######")

        # update positions
        dp = dict((k, v) for k, v in [(s, 0.0) for s in self.symbol_list])
        dp['datetime'] = bars[self.symbol_list[0]][0][1]  # 取第一个股票的第一条记录的第二栏位。索引字段
        #         [('000063',
        #   Timestamp('2017-07-21 00:00:00'),
        #   23.72,
        #   23.66,
        #   24.37,
        #   23.41,
        #   696489.0)]

        for s in self.symbol_list:
            dp[s] = self.current_positions[s]

        # append the current positions # 没有做任何修改呀，新增有意义吗？
        self.all_positions.append(dp)
        # update holdings
        dh = dict((k, v) for k, v in [(s, 0) for s in self.symbol_list])
        # dh = {symbol1:0,symbol1:0,...}
        dh['datetime'] = bars[self.symbol_list[0]][0][1]
        dh['cash'] = self.current_holdings['cash']
        dh['commission'] = self.current_holdings['commission']
        dh['total'] = self.current_holdings['cash']
        #         print(f"$$$$$$$$$$$:{dh}")

        # dh = {total:0,datetime:0,cash:0,commission:0,symbol1:0,symbol1:0,...}

        for s in self.symbol_list:
            market_value = self.current_positions[s] * bars[s][0][5]  # 该股票的市值=持有数*最新的收盘价
            dh[s] = market_value
            dh['total'] += market_value  # 这里只计算当前的最新总资产。主要是后面计算收益的时候用的就是每日的总资产 计算净利值曲线
        #             print(f"in for loop:{dh}")

        # append the current holdings
        #         print("$$$$$$$append the current holdings$$$$$$")
        print(dh)
        self.all_holdings.append(dh)

    #         print("update_timeindex return all_holdings:")
    #         for dit in self.all_holdings:
    #             print(dit)

    def update_positions_from_fill(self, fill):
        """
        Takes a FilltEvent object and updates the position matrix
        to reflect the new position.

        Parameters:
        fill - The FillEvent object to update the positions with.
        """
        # Check whether the fill is a buy or sell
        fill_dir = 0
        if fill.direction == 'BUY':
            fill_dir = 1
        if fill.direction == 'SELL':
            fill_dir = -1

        # Update positions list with new quantities 根据订单交易后产出的事件信号更新每只股票当前的持仓情况
        print("before update_positions_from_fill update current_positions:")
        for dit in self.all_holdings:
            print(dit)

        self.current_positions[fill.symbol] += fill_dir * fill.quantity

        print("after update_positions_from_fill update current_positions:")
        for dit in self.all_holdings:
            print(dit)

    def update_holdings_from_fill(self, fill):
        """
        Takes a FillEvent object and updates the holdings matrix
        to reflect the holdings value.

        Parameters:
        fill - The FillEvent object to update the holdings with.
        """
        # Check whether the fill is a buy or sell
        fill_dir = 0
        if fill.direction == 'BUY':
            fill_dir = 1
        if fill.direction == 'SELL':
            fill_dir = -1

        # Update holdings list with new quantities
        fill_cost = self.bars.get_latest_bars(fill.symbol)[0][5]  # Close price
        cost = fill_dir * fill_cost * fill.quantity  # 交易的金额 = 交易方向*当前收盘价格*交易数据量
        self.current_holdings[fill.symbol] += cost
        self.current_holdings['commission'] += fill.commission
        self.current_holdings['cash'] -= (cost + fill.commission)  # 可用资金 = 可用资金 - 交易金额 - 佣金
        self.current_holdings['total'] -= (cost + fill.commission)  # 总资产 = 总资产 - 交易金额 - 佣金 有点不对吧？ 卖就是+ 买的话也得+呀

    def update_fill(self, event):
        """ update the portfolio current positions and holding from a fillevent
        :param event:
        :return:
        """

        if event.type == 'FILL':
            self.update_holdings_from_fill(event)
            self.update_positions_from_fill(event)

    def generate_naive_order(self, signal):
        """
        Simply transacts an OrderEvent object as a constant quantity
        sizing of the signal object, without risk management or
        position sizing considerations.

        Parameters:
        signal - The SignalEvent signal information.
        """
        order = None

        symbol = signal.symbol
        direction = signal.signal_type
        strength = signal.strength

        mkt_quantity = floor(100 * strength)
        cur_quantity = self.current_positions[symbol]
        order_type = 'MKT'

        if direction == 'LONG' and cur_quantity == 0:
            order = OrderEvent(symbol, order_type, mkt_quantity, 'BUY')
        if direction == 'SHORT' and cur_quantity == 0:
            order = OrderEvent(symbol, order_type, mkt_quantity, 'SELL')
        if direction == 'EXIT' and cur_quantity > 0:
            order = OrderEvent(symbol, order_type, abs(cur_quantity), 'SELL')
        if direction == 'EXIT' and cur_quantity < 0:
            order = OrderEvent(symbol, order_type, abs(cur_quantity), 'BUY')

        return order

    def update_signal(self, event):
        """ acts on a signalEvent to generate new orders
        based on the portfolio logic
        :param event:
        :return:
        """

        if event.type == 'SIGNAL':
            order_event = self.generate_naive_order(event)
            self.events.put(order_event)

    def create_equity_curve_dataframe(self):
        """ creates a pandas dataframe from the all_holdings
        list of dictionaries.
        :return:
        """

        curve = pd.DataFrame(self.all_holdings)
        print(" create_equity_curve_dataframe update all_holdings:")
        #        dit = {'000063': 4235.0, 'datetime': Timestamp('2020-05-08 00:00:00'), 'cash': 95083.09999999999, 'commission': 3.9000000000000004, 'total': 99318.09999999999}
        #         for dit in self.all_holdings:
        #             print(dit)
        #         print(f"@@@1create_equity_curve_dataframe.curve:{curve.head(10)}")
        curve.set_index('datetime', inplace=True)
        curve['return'] = curve['total'].pct_change()
        #         print(f"@@@@@@:{curve.tail(10)}")
        curve['equity_curve'] = (1.0 + curve['return']).cumprod()

        self.equity_curve = curve

    #         print(f"@@@2create_equity_curve_dataframe.curve:{curve.head(10)}")

    def output_summary_stats(self):
        """
        Creates a list of summary statistics for the portfolio.
        """
        total_return = self.equity_curve['equity_curve'][-1]
        returns = self.equity_curve['return']
        pnl = self.equity_curve['equity_curve']
        # periods = 252 * 60 * 6.5

        earn_rate_year = (1 + np.mean(returns)) ** 252 - 1
        print("平均年化收益率：", earn_rate_year)

        sharpe_ratio = create_sharpe_ratio(returns, periods=252)
        max_dd, dd_duration = create_drawdowns(pnl)
        # drawdown, max_dd, dd_duration = create_drawdowns(pnl)
        #         # self.equity_curve['drawdown'] = drawdown

        stats = [("Total Return", "%0.2f%%" % ((total_return - 1.0) * 100.0)),
                 ("Avg Year Return", "%0.2f%%" % (earn_rate_year * 100.0)),
                 ("Sharpe Ratio", "%0.2f" % sharpe_ratio),
                 ("Max Drawdown", "%0.2f" % (max_dd)),
                 ("Drawdown Duration", "%d" % dd_duration)]

        filesavepath = os.path.join(self.savepath, 'equity.csv')
        self.equity_curve.to_csv(filesavepath, index=False)

        # 绘制累计日收益率曲线
        self.plot_equity_curve()
        return stats

    def plot_equity_curve(self):
        """ plot the equity curve
        :return:
        """
        plt.figure(figsize=(10, 8))
        self.equity_curve['equity_curve'].plot()
        plt.title("the equity curve:")
        plt.legend()
        filesavepath = os.path.join(self.savepath, 'equity_curve.png')

        plt.savefig(filesavepath)
        plt.show()


if __name__ == "__main__":
    print("portfolio_ is ok!!")

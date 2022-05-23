from itertools import combinations
from math import factorial
from random import shuffle

import numpy as np
import pandas as pd
import scipy.optimize as sco

from DataDownloader import Downloader

import os

class Optimizer:
    def __init__(self, available_money=100000, sector=None, share_limit=2, max_portion=.7, tickers=None,
                 currency='rub') -> None:
        """
        Takes in either a list of tickers or sector's name and currency and returns the most historically Sharpe-Optimal Portfolio.
        :param available_money: int [0:]
        :param sector: List[str]: One of ['it', 'consumer', 'health_care', 'financial', 'industrials', 'energy', 'other', 'utilities', 'materials', 'telecom', 'real_estate', 'ecomaterials', 'green_buildings', 'electrocars', 'green_energy']
        :param share_limit: int [2:] How many shares we'd like to see in our portfolio.
        :param max_portion: float [0:1]: How much portfolio space can a share take up 
        :param tickers: List[str]: A list of tickers to check.
        """
        self.results = {"fun": [], "x": [], "tickers": [], "profitability": []}
        self.DL = Downloader()
        self.COMBINATIONS_LIMIT = 2000
        self.currency = currency
        self.DATAFRAME_MAIN = self.load_data(tickers, sector)
        self.share_limit = share_limit
        self.M_avl = available_money
        self.CURRENT_COMBINATION = []
        self.max_portion = max_portion

    def compute(self):
        combinations_to_check = self._compute_combinations(len(self.tickers), self.share_limit)
        tickers_to_compute = [self.tickers]
        n = 1
        if combinations_to_check > self.COMBINATIONS_LIMIT:
            tickers_to_compute, n = self.allocate_tickers()
        for _ in range(n):
            for ticker_combination in tickers_to_compute:
                print(ticker_combination)
                self.compute_on_batch(ticker_combination)
            tickers_to_compute, _ = self.allocate_tickers()
        self.produce_output()
        return self.results

    def produce_output(self):
        temp = {}
        column_id_map = {col: self.DATAFRAME_MAIN.columns.get_loc(col) for col in self.DATAFRAME_MAIN.columns}

        r = pd.DataFrame().from_dict(self.results).sort_values(['fun'], ascending=False)
        self.results = r.to_dict(orient='list')

        for w, ti in zip(self.results['x'][-1], self.results['tickers'][-1]):
            temp[ti] = [(self.M_avl / self.DATAFRAME_MAIN.iloc[-1, column_id_map[ti]] * w).round(0), (self.M_avl / self.DATAFRAME_MAIN.iloc[-1, column_id_map[ti]] * w).round(0) * self.DATAFRAME_MAIN.iloc[-1, column_id_map[ti]]]
        print("\n"*3 + "*"*25)
        os.system("cls")
        print(f"Наилучший результат значения Шарпа - {self.results['fun'][0]}\nОн достигается следующим образом:")
        for k in temp.keys():
            print(f"Купить {int(temp[k][0])} акций {k} за {int(temp[k][1])} {self.currency}")

    def allocate_tickers(self):
        """
        Shuffle the list of tickers and split it into n parts.
        :return:
            _s: The split list of format List[List[str]]
            n: How many splits there were. Also used to measure how many more such splits we'd need to process.
        """
        shuffle(self.tickers)
        for n in range(2, 10):
            _s = np.array_split(self.tickers, n)
            _r = self._compute_combinations(max([len(c) for c in _s]), self.share_limit)
            if _r < self.COMBINATIONS_LIMIT:
                return _s, n

    def compute_on_batch(self, tickers):
        """
        Minimize the target function using a portion of the tickers and append the results to the results list.
        :param tickers: The portion of tickers, all combinations of which will get checked and solved.
        :return: None
        """
        print(f"Computing the best Sharpe Ratio for the following tickers: {tickers}")
        constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1})  # Constraints
        bounds = tuple((0, self.max_portion) for _ in range(self.share_limit))  # Bounds
        same_values = np.array(
            self.share_limit * [1. / self.share_limit, ])  # Equalized values to pass to the optimizer
        for ticker_combination in combinations(tickers, self.share_limit):
            self.CURRENT_COMBINATION = ticker_combination
            res = sco.minimize(self.target_function, same_values, constraints=constraints, bounds=bounds,
                               method="SLSQP")
            if -res['fun'].round(5) not in self.results['fun']:
                self.results['fun'].append(-res['fun'].round(5))
                self.results['x'].append(res['x'].round(3))
                self.results['tickers'].append(ticker_combination)
                self.results['profitability'].append(
                    (np.sum(self._compute_log_returns(list(ticker_combination)).mean() * res['x']) * 252).round(3))

    def target_function(self, portions):
        """
        The target function which needs to get minimized. The lower the negative Sharpe ratio, the better.
        :param portions:
        :return
        """
        _log_ret = self._compute_log_returns(self.CURRENT_COMBINATION)
        _hist_ret = np.sum(_log_ret.mean() * portions) * 252
        _hist_vol = np.sqrt(np.dot(portions.T, np.dot(_log_ret.cov() * 252, portions)))
        return -(_hist_ret - 0.05) / _hist_vol

    def load_data(self, tickers, sector):
        """
        Converts a list of tickers into Ticker to Open prices dataframe.

        Example:
        If tickers = ["SFTL" ,"POSI", "VKCO", "CIAN", "QIWI"]
        returns:
                      SFTL   POSI   VKCO   CIAN   QIWI
        2021-12-21  466.20  999.4  915.0  894.6  552.0
        2021-12-22  477.00  945.0  924.6  928.8  565.5
        2021-12-23  474.40  916.0  912.2  987.8  559.5
        2021-12-24  473.01  910.2  878.4  991.4  563.5
        2021-12-27  478.60  917.6  890.6  998.0  556.5
        ...            ...    ...    ...    ...    ...
        2022-05-17  168.40  960.2  358.0  330.0  321.0
        2022-05-18  160.50  963.6  362.8  338.0  329.0
        2022-05-19  164.90  954.0  379.4  342.0  330.0
        2022-05-20  156.40  952.0  363.0  343.8  364.0

        :param tickers: List[str] | None: Tickers, the data of which to use.
        :param sector: str | None: Sector to pull the tickers from.
        :return: _nd: pandas.DataFrame: Concatenated pandas array with:
            Columns - Ticker,
            Index - Date,
            values - Open prices for the Date
        """
        if not any((sector, tickers)):
            raise Exception("No ticker data detected.")
        if sector is not None:
            _t = self.DL.sectors[sector]
        else:
            _t = tickers
        _t = list(set(set(_t) & set(self.DL.currency[self.currency])))
        self.tickers = _t
        if len(_t) <= 1:
            raise Exception("Please Check your ticker selection. Currently, there's only one or less selected.")
        print(f"Starting to load data for the following tickers: {_t}")


        _rb = self.DL.get_data_batch(tickers=_t, days=365, candle_interval='1d')  # Raw batched data
        _dl = {k: pd.DataFrame().from_dict(data=v[0]).set_index("Date").iloc[:, 0] for k, v in zip(_rb.keys(), _rb.values())}  # Leave only Date and Open columns
        # Leave out the time part of the timedate

        for k in _dl.keys(): 
            try: _dl[k].index = _dl[k].index.date
            except AttributeError: pass

        _nd = pd.concat(_dl, axis=1).sort_index()  # Concatenate the dataframes into one
        print("Data loaded.")
        return _nd

    def _compute_combinations(self, N, R):
        """
        Computes the amount of combinations by which you could select R samples out of the whole N without
        any of the selections being the same.
        :param N: int: The number of the elements in the set
        :param R: int: The number of elements to pull from the set
        :return: The number of all possible combinations satisfying the above.
        """
        return factorial(N) / (factorial(N - R) * factorial(R))

    def _compute_log_returns(self, tickers):
        """
        Returns the log returns of a selection of tickers in
        :param tickers: List[str]
        :return:
        """
        _dfp = self.DATAFRAME_MAIN[list(tickers)]
        return np.log(_dfp / _dfp.shift()).fillna(0)

if __name__ == "__main__":
    tickers = None
    avaliable_funds = int(input("Пожалуйста, введите сумму, на которую Вам бы хотелось собрать портфель: \n>"))
    sector = input("Если Вы хотели бы проверить акции определенного сектора, введите его. Если нет, то пропустите этот шаг.\nДоступные секторы: it, consumer, health_care, financial, industrials, energy, other, utilities, materials, telecom, 'real_estate', 'ecomaterials', 'green_buildings', electrocars, green_energy\n>")
    if sector == '':
        sector = None
    if sector == None:
        tickers = input("Введите тикеры, которые Вам бы хотелось проверить. (Пример: YNDX,SBERP,RUAL): \n>")
        if tickers != '':
            tickers = tickers.split(',')
    currency = input("Валюта, за которую торгуется акция. (По умолчанию - rub): \n>")
    if currency == '':
        currency = 'rub'
    share_limit = input("Как много бумаг Вы бы хотли видеть в готовом портфеле? (Максимум бумаг на рассмотрение: 5): \n>")
    if share_limit != '':
        share_limit = int(share_limit)
    else:
        share_limit = 3
    res = Optimizer(available_money=avaliable_funds, sector=sector, tickers=tickers, currency=currency, share_limit=share_limit).compute()

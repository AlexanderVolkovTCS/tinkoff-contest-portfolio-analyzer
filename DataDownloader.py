import pathlib
import os
import datetime
import pickle
from tinkoff.invest import CandleInterval, Client
from tinkoff.invest.utils import now
from Log import Log
from config import (
    TOKEN,
    METADATA_EXPIRES,
    DAY_HISTORICAL_DATA_EXPIRES,
    HOUR_HISTORICAL_DATA_EXPIRES,
    FIFTEEN_MINUTES_HISTORICAL_DATA_EXPIRES,
    ONE_MINUTE_HISTORICAL_DATA_EXPIRES
)
from time import sleep


class Downloader:
    def __init__(self) -> None:
        self.metadata_dir = "Data"
        self.candles_directory = os.path.join(self.metadata_dir, "historical_data")
        self._init_dirs()
        self.metadata_fname = "metadata.pk"
        self.L = Log()
        self.client = Client(token=TOKEN)
        (
            self.ticker_to_figi_map,
            self.figi_to_ticker_map,
            self.sectors,
            self.currency
        ) = self._init_metadata()

    def get_historical_data_by(
        self, ticker=None, figi=None, candle_interval="1h", days=365, force_redownload=False
    ):
        assert any((ticker, figi))
        if ticker == None:
            if not figi in self.figi_to_ticker_map.keys():
                raise Exception(
                    "FIGI does not exist in the Metadata file. Please make sure you've specified the FIGI correctly. The program currently only supports shares and will NOT work with bonds, ETFs or any other financial instrument."
                )
            ticker = self.figi_to_ticker_map[figi]
        else:
            if not ticker in self.ticker_to_figi_map.keys():
                raise Exception(
                    "Ticker does not exist in the Metadata file. Please make sure you've specified the ticker correctly. The program currently only supports shares and will NOT work with bonds, ETFs or any other financial instrument."
                )
        if candle_interval not in self._CI.keys():
            raise Exception(
                f"Candle resolution should be one of: {list(self._CI.keys())}"
            )
        self.L(f"Starting to get historical data for {ticker}")
        FILENAME = f"{ticker}-{candle_interval}-{days}.pk"
        EXPIRES = [
            HOUR_HISTORICAL_DATA_EXPIRES,
            DAY_HISTORICAL_DATA_EXPIRES,
            FIFTEEN_MINUTES_HISTORICAL_DATA_EXPIRES,
            ONE_MINUTE_HISTORICAL_DATA_EXPIRES
        ][[x for x in self._CI.keys()].index(candle_interval)]

        candle_interval = self._CI[candle_interval]

        if force_redownload:
            return self._download_raw(ticker, days, candle_interval)

        if FILENAME in os.listdir(self.candles_directory):
            if not self._expired(
                os.path.join(self.candles_directory, FILENAME), EXPIRES
            ):
                self.L("Found existing and non-expired file. Returning.")
                return pickle.load(
                    open(os.path.join(self.candles_directory, FILENAME), "rb")
                ), 0
        self.L("No existing non-expired file was found. Initiating download.")
        raw = self._download_raw(ticker, days, candle_interval)
        pickle.dump(raw, open(os.path.join(self.candles_directory, FILENAME), "wb"))
        return raw, 1

    def get_data_batch(self, tickers, days, candle_interval):
        days_to_download = days * len(tickers)
        candles_to_download = int(days_to_download / self._POD[candle_interval])
        hard_limit_on_data_requests_per_minute = 15000
        candles_per_ticker = candles_to_download / len(tickers)
        if candles_per_ticker > hard_limit_on_data_requests_per_minute:
            raise Exception("Data requested is larger than specified limit.")
        data = {}
        self.L("Download started")
        if candles_to_download < hard_limit_on_data_requests_per_minute:
            self.L("Downloading the whole batch")
            for ticker in tickers:
                data[ticker] = self.get_historical_data_by(
                        ticker=ticker, candle_interval=candle_interval, days=days
                    )
                
        else:
            if candles_per_ticker > hard_limit_on_data_requests_per_minute / 2:
                for ticker in tickers:
                    data[ticker], dwn = self.get_historical_data_by(
                            ticker=ticker, candle_interval=candle_interval, days=days
                        )
                    sleep(60.0 * dwn)
            else:
                _c = 0
                for ticker in tickers:
                    if (_c > hard_limit_on_data_requests_per_minute - candles_per_ticker*2):
                        sleep(60.0)
                        _c = 0
                    data[ticker], dwn = self.get_historical_data_by(
                            ticker=ticker, candle_interval=candle_interval, days=days
                        )
                    
                    _c += dwn* candles_per_ticker
        return data

    def check_metadata(self):
        """
        A function that checks whether the required metadata exists and is not expired.
        """
        if self.metadata_fname in os.listdir(self.metadata_dir):
            self.L(f"Found {self.metadata_fname}. Checking for expiry.")
            if self._expired(
                os.path.join(self.metadata_dir, self.metadata_fname), METADATA_EXPIRES
            ):
                return 0
            else:
                return 1
        else:
            return 0

    def download_metadata(self):
        """
        Downloads metadata (FIGI to ticker and sector to ticker maps for convenience)
        """
        ticker_map = {}
        sector_map = {}
        currency_map = {}
        with self.client as client:
            share_data = client.instruments.shares()
        for share in share_data.instruments:
            ticker_map[share.ticker] = share.figi
            if share.sector not in sector_map.keys():
                sector_map[share.sector] = []
            sector_map[share.sector].append(share.ticker)
            if share.currency not in currency_map.keys():
                currency_map[share.currency] = []
            currency_map[share.currency].append(share.ticker)
        metadata = {"ticker_map": ticker_map, "sector_map": sector_map, "currency_map": currency_map}
        pickle.dump(
            metadata, open(os.path.join(self.metadata_dir, self.metadata_fname), "wb")
        )
        self.L("Metadata dumped.")
        return metadata

    def _init_metadata(self):
        "Download metadata if doesn't exist or is expired"
        data = {}
        if self.check_metadata():
            self.L("Loading metadata from disk")
            data = pickle.load(
                open(os.path.join(self.metadata_dir, self.metadata_fname), "rb")
            )
        else:
            self.L("Downloading metadata")
            data = self.download_metadata()

        return (
            data["ticker_map"],
            {v: k for k, v in data["ticker_map"].items()},
            data["sector_map"],
            data["currency_map"]
        )

    def _init_dirs(self):
        "Create required directories if none were found."
        if not os.path.exists(self.metadata_dir):
            os.mkdir(self.metadata_dir)
        if not os.path.exists(self.candles_directory):
            os.mkdir(self.candles_directory)

    def _expired(self, fname, delta_days):
        "Check whether a file needs to be updated."
        self.L(f"Checking {fname} for expiry.")
        MD_creation_datetime = datetime.datetime.fromtimestamp(
            pathlib.Path(fname).stat().st_ctime
        )
        timedelta = datetime.datetime.now() - MD_creation_datetime
        if timedelta.days > delta_days:
            return 1
        else:
            return 0

    def _download_raw(self, ticker, days, candle_interval):
        "Main logic for downloading the raw data."
        self.L(f"Downloading raw data for {ticker}")

        def process_quotation(quotation):
            whole = quotation.units
            rest = quotation.nano / (10**9)
            return whole + rest

        temp = {
            "Date": [],
            "Open": [],
            "High": [],
            "Low": [],
            "Close": [],
            "Volume": [],
        }

        with Client(TOKEN) as client:
            for candle in client.get_all_candles(
                figi=self.ticker_to_figi_map[ticker],
                from_=now() - datetime.timedelta(days=days),
                interval=candle_interval,
            ):
                temp["Date"].append(candle.time)
                temp["Close"].append(process_quotation(candle.close))
                temp["High"].append(process_quotation(candle.high))
                temp["Low"].append(process_quotation(candle.low))
                temp["Open"].append(process_quotation(candle.open))
                temp["Volume"].append(candle.volume)
        return temp

    @property
    def _CI(self):
        "String to CandleInterval map."
        return {
            "1h": CandleInterval.CANDLE_INTERVAL_HOUR,
            "1d": CandleInterval.CANDLE_INTERVAL_DAY,
            "15m": CandleInterval.CANDLE_INTERVAL_15_MIN,
            '1m': CandleInterval.CANDLE_INTERVAL_1_MIN
        }

    @property
    def _POD(self):
        "String to part of day map."
        return {"1h": 1 / 12, "1d": 1, "15m": 0.25 / 12, '1m': 1/60 / 12}


if __name__ == "__main__":
    d = Downloader()
    data = d.get_data_batch(["AAPL", "SBERP", "YNDX"], 1, candle_interval='1m')
    self.L(data)

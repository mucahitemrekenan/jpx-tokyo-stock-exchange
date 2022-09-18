import pandas as pd
import numpy as np
from decimal import ROUND_HALF_UP, Decimal
from tsfresh.feature_extraction.feature_calculators import autocorrelation, linear_trend, partial_autocorrelation
from sklearn.metrics import mean_squared_error, mean_absolute_error
import jpx_tokyo_market_prediction


# The function provided by the competition hosts. It generates AdjustedClose using AdjustmentFactor value. 
# This should reduce historical price gap caused by split/reverse-split.
def adjust_price(price):
    """
    Args:
        price (pd.DataFrame)  : pd.DataFrame include stock_price
    Returns:
        price DataFrame (pd.DataFrame): stock_price with generated AdjustedClose
    """
    # transform Date column into datetime
    price.loc[:, "Date"] = pd.to_datetime(price.loc[:, "Date"], format="%Y-%m-%d")

    def generate_adjusted_close(df):
        """
        Args:
            df (pd.DataFrame)  : stock_price for a single SecuritiesCode
        Returns:
            df (pd.DataFrame): stock_price with AdjustedClose for a single SecuritiesCode
        """
        # sort data to generate CumulativeAdjustmentFactor
        df = df.sort_values("Date", ascending=False)
        # generate CumulativeAdjustmentFactor
        df.loc[:, "CumulativeAdjustmentFactor"] = df["AdjustmentFactor"].cumprod()
        
        df.loc[:, "AdjustedClose"] = (df["CumulativeAdjustmentFactor"] * df["Close"]).map(
            lambda x: float(Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)))
        # reverse order
        df = df.sort_values("Date")
        # to fill AdjustedClose, replace 0 into np.nan
        df.loc[df["AdjustedClose"] == 0, "AdjustedClose"] = np.nan
        # forward fill AdjustedClose
        df.loc[:, "AdjustedClose"] = df.loc[:, "AdjustedClose"].ffill()
        return df
    # generate AdjustedClose
    price = price.sort_values(["SecuritiesCode", "Date"])
    price = price.groupby("SecuritiesCode").apply(
        generate_adjusted_close).reset_index(drop=True)
    return price


# This function also provided by competition hosts. It calculates model performances, 
# which determine leaderboard and competition winners. We call it competition metric.
def calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2) -> float:
    """
    Args:
        df (pd.DataFrame): predicted results
        portfolio_size (int): # of equities to buy/sell
        toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
    Returns:
        (float): sharpe ratio
    
    Working Principle:
        1) It creates 200 weights between 2 and 1 with linear space. 
        2) For purchase calculation, it sorts target values in ascending order takes the 200 highest,
        and multiplies with weights. Then sums those 200 values as 'S up'. 
        3) For short calculation, it sorts target values in descending order takes the 200 lowest,
        and multiply with weights. Then sum those 200 values as 'S down'.
        4) Returns difference of 'S up' and 'S down'. We call that spread return.
        5) At the end, it divides the mean of spread return value by the standard deviation of spread return value 
        and we get the Sharpe ratio.
    """
    def _calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):
        """
        Args:
            df (pd.DataFrame): predicted results
            portfolio_size (int): # of equities to buy/sell
            toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
        Returns:
            (float): spread return
        """
        assert df['Rank'].min() == 0
        assert df['Rank'].max() == len(df['Rank']) - 1
        weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
        purchase = (df.sort_values(by='Rank')['Target'][:portfolio_size] * weights).sum() / weights.mean()
        short = (df.sort_values(by='Rank', ascending=False)['Target'][:portfolio_size] * weights).sum() / weights.mean()
        return purchase - short

    buf = df.groupby('Date').apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    sharpe_ratio = buf.mean() / buf.std()
    return sharpe_ratio


# Statistical functions below pulled from tsfresh library which is very useful for time series data.
def autocorrelation_func(x):
    return np.nan_to_num(autocorrelation(x, 1))


def linear_trend_func(x):
    return linear_trend(x, [{"attr": "slope"}])[0][1]


def partial_autocorrelation_func(x):
    return list(partial_autocorrelation(x, [{'lag': 1}]))[0][1]


def std_func(x):
    return np.std(x)


# Feature creators based on tsfresh functions
def create_return_features(data, frames):
    for frame in frames:
        data["Return_{}Day".format(frame)] = data.groupby("SecuritiesCode")['AdjustedClose'].pct_change(frame)


def create_volatility_features(data, frames):
    for frame in frames:
        data["Volatility_{}Day".format(frame)] = np.log(data['AdjustedClose']).groupby(data["SecuritiesCode"])\
            .diff().rolling(frame).std()
    
    
def create_autocorr_features(data, frames):
    for frame in frames:
        data['AutoCorrelation_{}Day'.format(frame)] = data.groupby("SecuritiesCode")['AdjustedClose']\
            .rolling(window=frame).apply(autocorrelation_func).values


def create_linear_trend_features(data, frames):
    for frame in frames:
        data['LinearTrend_{}Day'.format(frame)] = data.groupby("SecuritiesCode")['AdjustedClose']\
            .rolling(window=frame).apply(linear_trend_func).values


def create_partial_autocorr_features(data,frames):
    for frame in frames:
        data['PartialAutoCorrelation_{}Day'.format(frame)] = data.groupby("SecuritiesCode")['AdjustedClose']\
            .rolling(window=frame).apply(partial_autocorrelation_func).values


# Update fuctions
def update_printing_date_params(printing_params, train_start_date, train_end_date, valid_start_date, 
                                valid_end_date, x_train_shape):
    printing_params['train_start_date'] = train_start_date
    printing_params['train_end_date'] = train_end_date
    printing_params['valid_start_date'] = valid_start_date
    printing_params['valid_end_date'] = valid_end_date
    printing_params['x_train_shape'] = x_train_shape


def update_error_params(printing_params, rmse, mae, sharpe):
    printing_params['rmse'] = rmse
    printing_params['mae'] = mae
    printing_params['valid_sharpe'] = sharpe


def update_feat_imp_and_sharpe(feat_importance, sharpe_data, gbm, fold, columns, sharpe):
    feat_importance["Importance_Fold" + str(fold)] = gbm.feature_importances_
    feat_importance.set_index(columns, inplace=True)
    sharpe_data.append(sharpe)


def print_info(printing_params):
    print("\n========================== Fold {} ==========================".format(printing_params['fold'] + 1))
    print("Train Date range: {} to {}".format(printing_params['train_start_date'], printing_params['train_end_date']))
    print('Train Shape', printing_params['x_train_shape'])
    print("Valid Date range: {} to {}".format(printing_params['validation_start_date'],
                                              printing_params['validation_end_date']))
    print("Valid Sharpe: {}, RMSE: {}, MAE: {}".format(printing_params['sharpe'], printing_params['rmse'], 
                                                       printing_params['mae']))


def modeling(model, x_train, y_train, x_val, y_val):
    regressor = model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)],
                  verbose=0, eval_metric=['mae', 'mse'])
    y_pred = regressor.predict(x_val)
    return y_pred, regressor


def calculate_error_rates(y_val, y_pred, data):
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    sharpe = calc_spread_return_sharpe(data)
    return rmse, mae, sharpe


# Submission part
def make_submission(gbm, training_features):
    env = jpx_tokyo_market_prediction.make_env()
    iter_test = env.iter_test()

    cols = ['Date', 'SecuritiesCode', 'Open', 'High', 'Low', 'Close', 'Volume', 'AdjustmentFactor']
    train = train[train.Date >= '2021-08-01'][cols]

    counter = 0
    for (prices, _, _, _, _, sample_prediction) in iter_test:
        # We initialize raw_price data frame if we are in the first iteration.
        current_date = prices["Date"].iloc[0]
        if counter == 0:
            raw_price = train.loc[train["Date"] < current_date]
        raw_price = pd.concat([raw_price, prices[cols]]).reset_index(drop=True)

        price_data = adjust_price(raw_price)

        # Feature creation part
        create_return_features(price_data, [3, 4, 6, 8, 12, 18, 20, 30, 50])
        create_volatility_features(price_data, [3, 6, 8, 12, 20])
        create_autocorr_features(price_data, [4, 8, 12])
        create_linear_trend_features(price_data, [3, 6, 13, 16])
        create_partial_autocorr_features(price_data, [20, 30, 50])

        feature_data = price_data[price_data.Date == current_date][training_features]
        feature_data["pred"] = gbm.predict(feature_data)
        feature_data["Rank"] = (feature_data["pred"].rank(method="first", ascending=False) - 1).astype(int)

        sample_prediction["Rank"] = feature_data["Rank"].values

        # We check min max values and whether predicted all correctly.
        assert sample_prediction["Rank"].notna().all()
        assert sample_prediction["Rank"].min() == 0
        assert sample_prediction["Rank"].max() == len(sample_prediction["Rank"]) - 1

        env.predict(sample_prediction)
        counter += 1



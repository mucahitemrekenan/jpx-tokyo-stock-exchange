import warnings, gc
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
from decimal import ROUND_HALF_UP, Decimal
import time
from tqdm import tqdm
from seglearn.feature_functions import mean_diff
from tsfresh.feature_extraction.feature_calculators import autocorrelation, linear_trend, partial_autocorrelation
warnings.filterwarnings("ignore")
import sys
sys.path.append('../jpx-tokyo-stock-exchange')
from env import TRAIN_PATH, STOCK_PATH


train = pd.read_csv(TRAIN_PATH, parse_dates=['Date'])
stock_list = pd.read_csv(STOCK_PATH)

stock_list['SectorName'] = [i.rstrip().lower().capitalize() for i in stock_list['17SectorName']]
stock_list['Name'] = [i.rstrip().lower().capitalize() for i in stock_list['Name']]


def adjust_price(price):
    price.loc[:, "Date"] = pd.to_datetime(price.loc[:, "Date"], format="%Y-%m-%d")
    def generate_adjusted_close(df):
        df = df.sort_values("Date", ascending=False)
        df.loc[:, "CumulativeAdjustmentFactor"] = df["AdjustmentFactor"].cumprod()
        df.loc[:, "AdjustedClose"] = (df["CumulativeAdjustmentFactor"] * df["Close"]).map(lambda x: float(Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)))
        df = df.sort_values("Date")
        df.loc[df["AdjustedClose"] == 0, "AdjustedClose"] = np.nan
        df.loc[:, "AdjustedClose"] = df.loc[:, "AdjustedClose"].ffill()
        return df

    price = price.sort_values(["SecuritiesCode", "Date"])
    price = price.groupby("SecuritiesCode").apply(generate_adjusted_close).reset_index(drop=True)
    return price


train = train.drop('ExpectedDividend', axis=1).fillna(0)
prices = adjust_price(train)



def autocorrelation_func(x):
    return np.nan_to_num(autocorrelation(x, 1))

def autocorrelation_edge_func(x):
    return np.nan_to_num(autocorrelation(x, len(x) - 1))

def linear_trend_func(x):
    return linear_trend(x, [{"attr": "slope"}])[0][1]

def mean_diff_func(x):
    return mean_diff([x])[0]

def partial_autocorrelation_func(x):
    return list(partial_autocorrelation(x, [{'lag': 1}]))[0][1]

def std_func(x):
    return np.std(x)


def create_features(df):
    col = 'AdjustedClose'
    periods = [3, 4, 6, 8, 12, 18, 20, 30, 50]
    for period in tqdm(periods):
        start = time.time()
        df["Return_{}Day".format(period)] = df.groupby("SecuritiesCode")[col].pct_change(period)
        end = time.time()
        print("Return_{}Day".format(period), start - end)
        start = time.time()
        df["MovingAvg_{}Day".format(period)] = df.groupby("SecuritiesCode")[col].rolling(window=period).mean().values
        end = time.time()
        print("MovingAvg_{}Day".format(period), start - end)
        start = time.time()
        df["ExpMovingAvg_{}Day".format(period)] = df.groupby("SecuritiesCode")[col].ewm(span=period, adjust=False).mean().values
        end = time.time()
        print("ExpMovingAvg_{}Day".format(period), start - end)
        start = time.time()
        df["Volatility_{}Day".format(period)] = np.log(df[col]).groupby(df["SecuritiesCode"]).diff().rolling(period).std()
        end = time.time()
        print("Volatility_{}Day".format(period), start - end)
        start = time.time()
        df['AutoCorrelation_{}Day'.format(period)] = df.groupby("SecuritiesCode")[col].rolling(window=period).apply(autocorrelation_func).values
        end = time.time()
        print("AutoCorrelation_{}Day".format(period), start - end)
        start = time.time()
        df['LinearTrend_{}Day'.format(period)] = df.groupby("SecuritiesCode")[col].rolling(window=period).apply(linear_trend_func).values
        end = time.time()
        print("LinearTrend_{}Day".format(period), start - end)
        start = time.time()
        df['MeanDiff_{}Day'.format(period)] = df.groupby("SecuritiesCode")[col].rolling(window=period).apply(mean_diff_func).values
        end = time.time()
        print("MeanDiff_{}Day".format(period), start - end)
        start = time.time()
        df['PartialAutoCorrelation_{}Day'.format(period)] = df.groupby("SecuritiesCode")[col].rolling(window=period).apply(partial_autocorrelation_func).values
        end = time.time()
        print("PartialAutoCorrelation_{}Day".format(period), start - end)
        start = time.time()
        df['Std_{}Day'.format(period)] = df.groupby("SecuritiesCode")[col].rolling(window=period).apply(std_func).values
        end = time.time()
        print("Std_{}Day".format(period), start - end)
    return df

price_features = create_features(df=prices.copy())
price_features.drop(['RowId', 'SupervisionFlag', 'AdjustmentFactor', 'CumulativeAdjustmentFactor', 'Close'], axis=1, inplace=True)


def calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2) -> float:
    def _calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):
        assert df['Rank'].min() == 0
        assert df['Rank'].max() == len(df['Rank']) - 1
        weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
        purchase = (df.sort_values(by='Rank')['Target'][:portfolio_size] * weights).sum() / weights.mean()
        short = (df.sort_values(by='Rank', ascending=False)['Target'][:portfolio_size] * weights).sum() / weights.mean()
        return purchase - short
    buf = df.groupby('Date').apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    sharpe_ratio = buf.mean() / buf.std()
    return sharpe_ratio

drop_cols = ['PartialAutoCorrelation_3Day']

ts_fold = TimeSeriesSplit(n_splits=10, gap=10000)
prices = price_features.drop(columns=drop_cols).dropna().sort_values(['Date', 'SecuritiesCode'])
y = prices['Target'].to_numpy()
X = prices.drop(['Target'], axis=1)

feat_importance = pd.DataFrame()
sharpe_ratio = []

for fold, (train_idx, val_idx) in enumerate(ts_fold.split(X, y)):

    print("\n========================== Fold {} ==========================".format(fold + 1))
    X_train, y_train = X.iloc[train_idx, :], y[train_idx]
    X_valid, y_val = X.iloc[val_idx, :], y[val_idx]

    print("Train Date range: {} to {}".format(X_train.Date.min(), X_train.Date.max()))
    print('Train Shape', X_train.shape)
    print("Valid Date range: {} to {}".format(X_valid.Date.min(), X_valid.Date.max()))

    X_train.drop(['Date', 'SecuritiesCode'], axis=1, inplace=True)
    X_val = X_valid[X_valid.columns[~X_valid.columns.isin(['Date', 'SecuritiesCode'])]]
    val_dates = X_valid.Date.unique()[1:-1]

    params = {'n_estimators': 500,
              'num_leaves': 100,
              'learning_rate': 0.1,
              'colsample_bytree': 0.9,
              'subsample': 0.8,
              'reg_alpha': 0.4,
              'metric': 'mae',
              'random_state': 21}

    gbm = LGBMRegressor(**params).fit(X_train, y_train,
                                      eval_set=[(X_train, y_train), (X_val, y_val)],
                                      verbose=0,
                                      eval_metric=['mae', 'mse'])
    y_pred = gbm.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    feat_importance["Importance_Fold" + str(fold)] = gbm.feature_importances_
    feat_importance.set_index(X_train.columns, inplace=True)

    rank = []
    X_val_df = X_valid[X_valid.Date.isin(val_dates)]
    for i in X_val_df.Date.unique():
        temp_df = X_val_df[X_val_df.Date == i].drop(['Date', 'SecuritiesCode'], axis=1)
        temp_df["pred"] = gbm.predict(temp_df)
        temp_df["Rank"] = (temp_df["pred"].rank(method="first", ascending=False) - 1).astype(int)
        rank.append(temp_df["Rank"].values)

    stock_rank = pd.Series([x for y in rank for x in y], name="Rank")
    df = pd.concat([X_val_df.reset_index(drop=True), stock_rank,
                    prices[prices.Date.isin(val_dates)]['Target'].reset_index(drop=True)], axis=1)
    sharpe = calc_spread_return_sharpe(df)
    sharpe_ratio.append(sharpe)
    print("Valid Sharpe: {}, RMSE: {}, MAE: {}".format(sharpe, rmse, mae))

    del X_train, y_train, X_val, y_val
    gc.collect()

print("\nAverage cross-validation Sharpe Ratio: {:.4f}, standard deviation = {:.2f}.".format(np.mean(sharpe_ratio),
                                                                                             np.std(sharpe_ratio)))

feat_importance['avg'] = feat_importance.mean(axis=1)
feat_importance = feat_importance.sort_values(by='avg', ascending=True)

cols_fin = feat_importance.avg.nlargest(10).index.tolist()
cols_fin.extend(('Open', 'High', 'Low'))
X_train = prices[cols_fin]
y_train = prices['Target']
gbm = LGBMRegressor(**params).fit(X_train, y_train)



# Submission part
import jpx_tokyo_market_prediction

env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()

cols = ['Date', 'SecuritiesCode', 'Open', 'High', 'Low', 'Close', 'Volume', 'AdjustmentFactor']
train = train[train.Date >= '2021-08-01'][cols]

counter = 0
for (prices, options, financials, trades, secondary_prices, sample_prediction) in iter_test:

    current_date = prices["Date"].iloc[0]
    if counter == 0:
        df_price_raw = train.loc[train["Date"] < current_date]
    df_price_raw = pd.concat([df_price_raw, prices[cols]]).reset_index(drop=True)
    df_price = adjust_price(df_price_raw)
    features = create_features(df=df_price)
    feat = features[features.Date == current_date][cols_fin]
    feat["pred"] = gbm.predict(feat)
    feat["Rank"] = (feat["pred"].rank(method="first", ascending=False) - 1).astype(int)
    sample_prediction["Rank"] = feat["Rank"].values

    assert sample_prediction["Rank"].notna().all()
    assert sample_prediction["Rank"].min() == 0
    assert sample_prediction["Rank"].max() == len(sample_prediction["Rank"]) - 1

    env.predict(sample_prediction)
    counter += 1

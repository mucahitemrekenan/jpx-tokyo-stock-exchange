import warnings, gc
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor
from decimal import ROUND_HALF_UP, Decimal
warnings.filterwarnings("ignore")
import sys
sys.path.append('../jpx-tokyo-stock-exchange')
from env import TRAIN_PATH, STOCK_PATH
from preprocess import *


# Input files
train = pd.read_csv(TRAIN_PATH, parse_dates=['Date'])
stock_list = pd.read_csv(STOCK_PATH)

stock_list['SectorName'] = [i.rstrip().lower().capitalize() for i in stock_list['17SectorName']]
stock_list['Name'] = [i.rstrip().lower().capitalize() for i in stock_list['Name']]

train = train.drop('ExpectedDividend', axis=1).fillna(0)
prices = adjust_price(train)

# Because of my function structure I need to copy prices data frame as price_feature_data
# Otherwise functions override prices data frame, which i need in the modeling part.
price_feature_data = prices.copy()

# Feature creation part
create_return_features(price_feature_data, [3, 4, 6, 8, 12, 18, 20, 30, 50])
create_volatility_features(price_feature_data, [3, 6, 8, 12, 20])
create_autocorr_features(price_feature_data, [4, 8, 12])
create_linear_trend_features(price_feature_data, [3, 6, 13, 16])
create_partial_autocorr_features(price_feature_data, [20, 30, 50])

price_feature_data.drop(['RowId', 'SupervisionFlag', 'AdjustmentFactor', 'CumulativeAdjustmentFactor', 'Close'], 
                    axis=1, inplace=True)

ts_fold = TimeSeriesSplit(n_splits=10, gap=10000)
prices = price_feature_data.dropna().sort_values(['Date', 'SecuritiesCode'])
y = prices['Target'].to_numpy()
X = prices.drop(['Target'], axis=1)

feat_imp = pd.DataFrame()
sharpe_data = []
printing_params = dict()
model_params = {'n_estimators': 500,
                'num_leaves': 100,
                'learning_rate': 0.1,
                'colsample_bytree': 0.9,
                'subsample': 0.8,
                'reg_alpha': 0.4,
                'metric': 'mae',
                'random_state': 21}

for fold, (train_idx, val_idx) in enumerate(ts_fold.split(X, y)):

    X_train, y_train = X.iloc[train_idx, :], y[train_idx]
    X_valid, y_val = X.iloc[val_idx, :], y[val_idx]

    
    update_printing_date_params(printing_params, X_train.Date.min(), X_train.Date.max(),
                                X_valid.Date.min(), X_valid.Date.max(), X_train.shape)

    X_train.drop(['Date', 'SecuritiesCode'], axis=1, inplace=True)
    X_val = X_valid[X_valid.columns[~X_valid.columns.isin(['Date', 'SecuritiesCode'])]]
    val_dates = X_valid.Date.unique()[1:-1]

    y_pred, gbm = modeling(LGBMRegressor(**model_params))

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

    rmse, mae, sharpe = calculate_error_rates(y_val, y_pred, df)
    update_error_params(printing_params, rmse, mae, sharpe)
    update_feat_imp_and_sharpe(feat_imp, sharpe_data, gbm, fold, X_train.columns, sharpe)
    print_info(printing_params)

    del X_train, y_train, X_val, y_val
    gc.collect()

print("\nAverage cross-validation Sharpe Ratio: {:.4f}, standard deviation = {:.2f}.".format(np.mean(sharpe_data),
                                                                                             np.std(sharpe_data)))

feat_imp['avg'] = feat_imp.mean(axis=1)
feat_imp = feat_imp.sort_values(by='avg', ascending=True)

training_features = feat_imp.avg.nlargest(10).index.tolist()
training_features.extend(('Open', 'High', 'Low'))
X_train = prices[training_features]
y_train = prices['Target']
gbm = LGBMRegressor(**model_params).fit(X_train, y_train)

make_submission(gbm, training_features)
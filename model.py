#%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import norm, skew

pd.set_option("display.max_rows", 200)
pd.set_option('display.max_columns', 200)
train_df = pd.read_csv('./input/train.csv')
test_df = pd.read_csv('./input/test.csv')

train_df.head(10)

train_ID = train_df['Id']
test_ID = test_df['Id']
train_df.drop("Id", axis = 1, inplace = True)
test_df.drop("Id", axis = 1, inplace = True)

standard_df = ((train_df['SalePrice'] - train_df['SalePrice'].mean())/train_df['SalePrice'].std()).to_frame()

train_df['OutlierSalePrice'] = standard_df['SalePrice'].apply(lambda x: -2 < x < 4)
train_df = train_df[train_df['OutlierSalePrice'] == True].reset_index(drop=True)
train_df = train_df.drop('OutlierSalePrice', axis=1)

X = train_df.loc[:, train_df.columns != 'SalePrice']
y = train_df.loc[:, 'SalePrice']

train_df.corr()['SalePrice'].sort_values(ascending=False)

standard_df.hist(bins=50)
train_df[['SalePrice']].hist(bins=50)

# train_df.hist(bins=50)
# test_df.hist(bins=50)
# train_df.loc[:, 'MultiFullBathFlg'] = train_df['FullBath'].apply(lambda x: 1 if x > 1.0 else 0)
# test_df.loc[:, 'MultiFullBathFlg'] = test_df['FullBath'].apply(lambda x: 1 if x > 1.0 else 0)

ntrain = train_df.shape[0]
all_data = pd.concat([X, test_df], sort=False).reset_index(drop=True)
all_data
# Adding total sqfootage feature
missing_columns = ['FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature', 'Alley']
for column in missing_columns:
    all_data[column] = all_data[column].fillna("None")

def get_high_corr_columns(missing_column, df):
    high_corr_columns = df.corr()[[missing_column]].sort_values(by=missing_column,
                                    ascending=False).iloc[1:3,0].index.tolist()

    return high_corr_columns

def fill_missing_values(X, target_column, corr_columns):
    reg = lm.LinearRegression()
    indexer = X[target_column].isnull()
    if indexer.sum() > 0:
        reg.fit(X.loc[~indexer, corr_columns], X.loc[~indexer, target_column])
        predicted = reg.predict(X.loc[indexer, corr_columns])
        X.loc[indexer, target_column] = predicted

    return X

def preprocess(X, high_corr_columns, missing_columns):

    for i, missing_column in enumerate(missing_columns):
        X = fill_missing_values(X, missing_column, high_corr_columns[i])

    return X

def encode_category_val(X):
    category_columns = X.select_dtypes(include=[object]).columns.tolist()
    for column in category_columns:
        labels, uniques = pd.factorize(X[column])
        X[column] = labels
        # le = preprocessing.LabelEncoder()
        # le.fit(X[column])
        # label_encoded_column = le.transform(X[column])
        # X = X.copy()
        # X.loc[:, column] = pd.Series(label_encoded_column).astype('int')

    return X

#Validation function
n_folds = 5

def rmsle_cv(model, n_folds, X, y):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X.values)
    rmse= np.sqrt(-cross_val_score(model, X.values, y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

missing_columns2 = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
high_corr_columns = [get_high_corr_columns(m, all_data) for m in missing_columns2]

missing_columns3 = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
#BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBathも0で埋める
for col in missing_columns3:
    all_data[col] = all_data[col].fillna(0)

all_data = preprocess(all_data, high_corr_columns, missing_columns2)
all_data = pd.get_dummies(all_data)

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
left_time = 55 - (2018 - all_data['YearBuilt'])
left_time = left_time.apply(lambda x: 1 if x < 1 else x)
all_data['QualLefttime'] = all_data['OverallQual'] * left_time
all_data['TotalSFLefttime'] = all_data['TotalSF'] * left_time
all_data['TotalSFQualLefttime'] = all_data['OverallQual'] * all_data['TotalSF'] * left_time
all_data["TotalSFOverallQual"] = all_data["TotalSF"] * all_data["OverallQual"]

all_data['TotalRoomNum'] = all_data['TotRmsAbvGrd'] + all_data['BsmtFullBath'] + \
                            all_data['BsmtHalfBath'] + all_data['FullBath'] + \
                            all_data['HalfBath']
all_data['TotalRoomNumLefttime'] = all_data['TotalRoomNum'] * left_time

[c for c in all_data.columns if 'Fire' in c]
all_data['FireplaceQu_Ex']
all_data[['Fireplaces','FireplaceQu_Ex','FireplaceQu_Fa','FireplaceQu_Gd','FireplaceQu_None','FireplaceQu_Po','FireplaceQu_TA']].hist()
all_data[['Foundation_BrkTil', 'Foundation_CBlock', 'Foundation_PConc', 'Foundation_Slab', 'Foundation_Stone', 'Foundation_Wood']].hist()
remodel_left_time = 55 - (2018 - all_data['YearRemodAdd'])
remodel_left_time = remodel_left_time.apply(lambda x: 1 if x < 1 else x)
all_data['QualRemodLefttime'] = all_data['OverallQual'] * remodel_left_time
all_data['TotalSFRemodLefttime'] = all_data['TotalSF'] * remodel_left_time
all_data['TotalSFQualRemodLefttime'] = all_data['OverallQual'] * all_data['TotalSF'] * remodel_left_time

all_data['TotalRoomNumRemodLefttime'] = all_data['TotalRoomNum'] * remodel_left_time
all_data['BuiltLehmanFlg'] = all_data['YearBuilt'].apply(lambda x: 1 if 2005 < x < 2011 else 0)
all_data['SoldLehmanFlg'] = all_data['YrSold'].apply(lambda x: 1 if x > 2007 else 0)

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
numeric_feats
# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
all_data['Exterior2nd_Other']
skewness.head(15)

# X_train = preprocess(X_train, high_corr_columns, missing_columns2)
# X_val = preprocess(X_val, high_corr_columns, missing_columns2)
# X_test = preprocess(X_test, high_corr_columns, missing_columns2)

X_train, X_val, y_train, y_val = train_test_split(all_data[:ntrain], y, test_size=0.3, train_size=0.7)
X_train = X_train.reset_index(drop=True)
X_val = X_val.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_val = y_val.reset_index(drop=True)

# category_columns = X_train.select_dtypes(include=[object]).columns.tolist()
# lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=list(category_columns))
# lgb_val = lgb.Dataset(X_val, y_val, categorical_feature=list(category_columns))

# category_columns
# for column in category_columns:
#     X_train[column] = pd.Categorical(X_train[column])
#     X_val[column] = pd.Categorical(X_val[column])
#     X_test[column] = pd.Categorical(X_test[column])


model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=1000,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
clf = XGBRegressor()
score_lightgbm = rmsle_cv(model_lgb, 5, all_data[:ntrain], y)
score_xgboost = rmsle_cv(clf, 5, all_data[:ntrain], y)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score_lightgbm.mean(),
                                              score_lightgbm.std()))
print("XGBOOST score: {:.4f} ({:.4f})\n" .format(score_xgboost.mean(),
                                              score_xgboost.std()))
score_lightgbm
score_xgboost

GBoost = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)
score_gbdt = rmsle_cv(GBoost, 5, all_data[:ntrain], y)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score_gbdt.mean(), score_gbdt.std()))
score_gbdt
np.mean(score_lightgbm.tolist()+score_xgboost.tolist()+score_gbdt.tolist())

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.1, l1_ratio=.9, random_state=3))
score_elasticnet = rmsle_cv(ENet, 5, all_data[:ntrain], y)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score_elasticnet.mean(), score_elasticnet.std()))
score_elasticnet
score_gbdt
score_lightgbm
score_xgboost



numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(15)model = model_lgb.fit(all_data[:ntrain], y)

# all_data.drop([col for col, f in zip(all_data.columns, model.feature_importances_) if f < 5], axis=1, inplace=True)
#
# score = rmsle_cv(model_lgb, 5, all_data[:ntrain], y)
# print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

X_test = all_data[ntrain:]
model = model_lgb.fit(all_data[:ntrain], y)
y_pred = model.predict(X_test)

sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = y_pred
sub.to_csv('submission.csv',index=False)


train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=list(category_columns))
valid_data = lgb.Dataset(X_val, label=y_val, categorical_feature=list(category_columns))

parameters = {
    'application': 'regression',
    'metric': 'rmse',
    'boosting': 'gbdt',
    'num_leaves': 5,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'feature_fraction': 0.2319,
    'learning_rate': 0.05}


evals_result = {}
model = lgb.train(parameters,
                       train_data,
                       valid_sets=valid_data,
                       num_boost_round=720,
                       early_stopping_rounds=20,
                       evals_result=evals_result,
                       verbose_eval=20)

y_val_pred = model.predict(X_val)
y_pred = model.predict(X_test)

result_df = pd.Series(y_pred).to_frame()
result_df.index = np.arange(1461,1461+len(y_pred))
result_df = result_df.rename(columns={0: 'SalePrice'})
result_df.to_csv('result.csv', index=True, index_label='Id')

ax = lgb.plot_metric(evals_result, metric='rmse')
plt.show()

ax = lgb.plot_importance(model, max_num_features=20)
plt.show()

# X_train = X_train.fillna('')
# X_val = X_val.fillna('')
# X_test = X_test.fillna('')

# X_train = encode_category_val(X_train)
# X_val = encode_category_val(X_val)
# X_test = encode_category_val(X_test)

# float_columns = X_train.select_dtypes(include=[float]).columns.tolist()
# X_train[float_columns] = X_train[float_columns].astype(int)
# X_val[float_columns] = X_val[float_columns].astype(int)
# X_test[float_columns] = X_test[float_columns].astype(int)

# clf = XGBRegressor().fit(X_train, y_train)

# lgb_train = lgb.Dataset(X_train, y_train)
# lgb_eval = lgb.Dataset(X_val, y_val,reference=lgb_train)
#
# params = {
#     'task': 'train',
#     'boosting_type': 'gbdt',
#     'objective': 'regression',
#     'metric': {'l2'},
#     'num_leaves': 200,
#     'learning_rate': 0.003,
#     'num_iterations':100,
#     'feature_fraction': 0.52,
#     'bagging_fraction': 0.79,
#     'bagging_freq': 7,
#     'verbose': 0
# }
#
# gbm = lgb.train(params,
#                 lgb_train,
#                 num_boost_round=5000,
#                 valid_sets=lgb_eval,
#                 early_stopping_rounds=1000)
# X_pred= gbm.predict(X_test, num_iteration=gbm.best_iteration)

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0}

print('Starting training...')
# feature_name and categorical_feature
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10,
                valid_sets=lgb_train,  # eval training data
                )

gbm = lgb.LGBMRegressor(num_leaves=50, learning_rate=0.05, n_estimators=200)
gbm.fit(X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='l2',
        early_stopping_rounds=5,
        categorical_feature=category_columns)

# scores = cross_val_score(gbm, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
scores = cross_val_score(model, train_data, valid_data, cv=5, scoring="neg_mean_squared_log_error")
rmse_scores = np.sqrt(-scores)
print('The rmse of Train prediction is:', rmse_scores.mean())

y_val_pred = gbm.predict(lgb_val, num_iteration=gbm.best_iteration_)
print('The rmse of Test prediction is:', mean_squared_log_error(y_val, y_val_pred) ** 0.5)


y_test_pred = gbm.predict(lgb_test, num_iteration=gbm.best_iteration_)
# result_df = pd.Series(y_test_pred).to_frame()
#
# result_df.index = np.arange(1461,1461+len(y_pred))
# result_df = result_df.rename(columns={0: 'SalePrice'})
# result_df.to_csv('result.csv', index=True, index_label='Id')


pd.concat([X_train, y_train], axis=1).corr()['SalePrice'].sort_values(ascending=False)

result_df = pd.Series(y_pred).to_frame()
result_df.index = np.arange(1461,1461+len(y_pred))
result_df = result_df.rename(columns={0: 'SalePrice'})
result_df.to_csv('result.csv', index=True, index_label='Id')

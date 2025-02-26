# %  ---------------------------------------------------------------------------------------------------------------------------------------
# %  Hospitalization Models
# %  ---------------------------------------------------------------------------------------------------------------------------------------
# %	This code provides predictive model described in "Early prediction of level-of-care requirements in patients with COVID-19." - Elife(2020)
# %   
# %   Authors: Hao, Boran, Shahabeddin Sotudian, Taiyao Wang, Tingting Xu, Yang Hu, 
# %   Apostolos Gaitanidis, Kerry Breen, George C. Velmahos, and Ioannis Ch Paschalidis.
# %   
# % ---------------------------------------------------------------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
#  Data
# ------------------------------------------------------------------------------


import lightgbm as lgb
from sklearn.metrics import classification_report, f1_score, roc_curve, auc, accuracy_score
import pylab as pl
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn import preprocessing
import os
import math
import numpy as np
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_rows = None


# Load Preprocessed Data

Y = pd.read_csv('Final_Y.csv')
X = pd.read_csv('Final_X.csv')


# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

def stat_test(df, y):
    name = pd.DataFrame(df.columns, columns=['Variable'])
    df0 = df[y == 0]
    df1 = df[y == 1]
    pvalue = []
    y_corr = []
    for col in df.columns:
        if df[col].nunique() == 2:

            chi2stat, pval, Stat2chi = smprop.proportions_chisquare(
                [df0[col].sum(), df1[col].sum()], [len(df0[col]), len(df1[col])], value=None)
            pvalue.append(pval)

        else:
            pvalue.append(stats.ks_2samp(df0[col], df1[col]).pvalue)
        y_corr.append(df[col].corr(y))
    name['All_mean'] = df.mean().values
    name['y1_mean'] = df1.mean().values
    name['y0_mean'] = df0.mean().values
    name['All_std'] = df.std().values
    name['y1_std'] = df1.std().values
    name['y0_std'] = df0.std().values
    name['p-value'] = pvalue
    name['y_corr'] = y_corr
    # [['Variable','p-value','y_corr']]
    return name.sort_values(by=['p-value'])


def high_corr(df, thres=0.8):
    corr_matrix_raw = df.corr()
    corr_matrix = corr_matrix_raw.abs()
    high_corr_var_ = np.where(corr_matrix > thres)
    high_corr_var = [(corr_matrix.index[x],
                      corr_matrix.columns[y],
                      corr_matrix_raw.iloc[x,
                                           y]) for x,
                     y in zip(*high_corr_var_) if x != y and x < y]
    return high_corr_var


def df_fillna(df):
    df_nullsum = df.isnull().sum()
    for col in df_nullsum[df_nullsum > 0].index:
        df[col + '_isnull'] = df[col].isnull()
        df[col] = df[col].fillna(df[col].median())
    return df


def df_drop(df_new, drop_cols):
    return df_new.drop(df_new.columns[df_new.columns.isin(drop_cols)], axis=1)


def clf_F1(best_C_grid, best_F1, best_F1std, classifier, X_train,
           y_train, C_grid, nFolds, silent=True, seed=2020):
    # global best_C_grid,best_F1, best_F1std
    results = cross_val_score(
        classifier,
        X_train,
        y_train,
        cv=StratifiedKFold(
            n_splits=nFolds,
            shuffle=True,
            random_state=seed),
        n_jobs=-1,
        scoring='f1')  # cross_validation.
    F1, F1std = results.mean(), results.std()
    if silent == False:
        print(C_grid, F1, F1std)
    if F1 > best_F1:
        best_C_grid = C_grid
        best_F1, best_F1std = F1, F1std
    return best_C_grid, best_F1, best_F1std


def my_RFE(df_new, col_y='Hospitalization', my_range=range(
        5, 60, 2), my_penalty='l1', my_C=0.01, cvFolds=5, step=1):
    F1_all_rfe = []
    Xraw = df_new.drop(col_y, axis=1).values
    y = df_new[col_y].values
    names = df_new.drop(col_y, axis=1).columns
    for n_select in my_range:

        X = Xraw
        clf = LogisticRegression(
            C=my_C,
            penalty=my_penalty,
            class_weight='balanced',
            solver='liblinear')  # tol=0.01,
        rfe = RFE(clf, n_select, step=step)
        rfe.fit(X, y.ravel())
        X = df_new.drop(col_y, axis=1).drop(
            names[rfe.ranking_ > 1], axis=1).values

        best_F1, best_F1std = 0.1, 0
        best_C_grid = 0
        for C_grid in [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]:
            clf = LogisticRegression(
                C=C_grid,
                class_weight='balanced',
                solver='liblinear')  # penalty=my_penalty,
            best_C_grid, best_F1, best_F1std = clf_F1(
                best_C_grid, best_F1, best_F1std, clf, X, y, C_grid, cvFolds)
        F1_all_rfe.append((n_select, best_F1, best_F1std))
    F1_all_rfe = pd.DataFrame(
        F1_all_rfe, index=my_range, columns=[
            'n_select', "best_F1", "best_F1std"])
    F1_all_rfe['F1_'] = F1_all_rfe['best_F1'] - F1_all_rfe['best_F1std']

    X = Xraw
    clf = LogisticRegression(
        C=my_C,
        penalty=my_penalty,
        class_weight='balanced',
        solver='liblinear')  # 0.
    rfe = RFE(
        clf, F1_all_rfe.loc[F1_all_rfe['F1_'].idxmax(), 'n_select'], step=step)
    rfe.fit(X, y.ravel())
    id_keep_1st = names[rfe.ranking_ == 1].values
    return id_keep_1st, F1_all_rfe


def my_train(X_train, y_train, model='LR', penalty='l1', cv=5,
             scoring='f1', class_weight='balanced', seed=2020):
    if model == 'SVM':
        svc = LinearSVC(
            penalty=penalty,
            class_weight=class_weight,
            dual=False,
            max_iter=10000)
        parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        gsearch = GridSearchCV(svc, parameters, cv=cv, scoring=scoring)
    elif model == 'LGB':
        param_grid = {
            'num_leaves': range(2, 15, 2),
            'n_estimators': [50, 100, 500, 1000],
            'colsample_bytree': [0.1, 0.3, 0.7, 0.9]

        }
        lgb_estimator = lgb.LGBMClassifier(
            boosting_type='gbdt',
            objective='binary',
            learning_rate=0.1,
            random_state=seed)  # eval_metric='auc' num_boost_round=2000,
        gsearch = GridSearchCV(
            estimator=lgb_estimator,
            param_grid=param_grid,
            cv=cv,
            n_jobs=-1,
            scoring=scoring)
    elif model == 'RF':
        rfc = RandomForestClassifier(
            random_state=seed,
            class_weight=class_weight,
            n_jobs=-1)
        param_grid = {
            'max_features': [0.05, 0.1, 0.3, 0.5, 0.7, 1],
            'n_estimators': [100, 500, 1000],
            'max_depth': range(2, 10, 1)

        }
        gsearch = GridSearchCV(
            estimator=rfc,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring)

    else:
        LR = LogisticRegression(
            penalty=penalty,
            class_weight=class_weight,
            solver='liblinear',
            random_state=seed)
        parameters = {'C': [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
        gsearch = GridSearchCV(LR, parameters, cv=cv, scoring=scoring)

    gsearch.fit(X_train, y_train)
    clf = gsearch.best_estimator_
    if model == 'LGB' or model == 'RF':
        print('Best parameters found by grid search are:', gsearch.best_params_)

    print('train set accuracy:', gsearch.best_score_)
    return clf


def cal_f1_scores(y, y_pred_score):
    fpr, tpr, thresholds = roc_curve(y, y_pred_score)
    thresholds = sorted(set(thresholds))
    metrics_all = []
    for thresh in thresholds:
        y_pred = np.array((y_pred_score > thresh))
        metrics_all.append(
            (thresh, auc(
                fpr, tpr), f1_score(
                y, y_pred, average='micro'), f1_score(
                y, y_pred, average='macro'), f1_score(
                    y, y_pred, average='weighted')))
    metrics_df = pd.DataFrame(
        metrics_all,
        columns=[
            'thresh',
            'tr AUC',
            'tr micro F1-score',
            'tr macro F1-score',
            'tr weighted F1-score'])
    return metrics_df.sort_values(
        by='tr weighted F1-score', ascending=False).head(1)  # ['thresh'].values[0]


def cal_f1_scores_te(y, y_pred_score, thresh):
    fpr, tpr, thresholds = roc_curve(y, y_pred_score)
    y_pred = np.array((y_pred_score > thresh))
    metrics_all = [
        (thresh, auc(
            fpr, tpr), f1_score(
            y, y_pred, average='micro'), f1_score(
                y, y_pred, average='macro'), f1_score(
                    y, y_pred, average='weighted'))]
    metrics_df = pd.DataFrame(
        metrics_all,
        columns=[
            'thresh',
            'AUC',
            'micro F1-score',
            'macro F1-score',
            'weighted F1-score'])
    return metrics_df


def my_test(X_train, xtest, y_train, ytest, clf,
            target_names, report=False, model='LR'):
    if model == 'SVM':
        ytrain_pred_score = clf.decision_function(X_train)
    else:
        ytrain_pred_score = clf.predict_proba(X_train)[:, 1]
    metrics_tr = cal_f1_scores(y_train, ytrain_pred_score)
    thres_opt = metrics_tr['thresh'].values[0]
    # ytest_pred=clf.predict(xtest)
    if model == 'SVM':
        ytest_pred_score = clf.decision_function(xtest)
    else:
        ytest_pred_score = clf.predict_proba(xtest)[:, 1]
    metrics_te = cal_f1_scores_te(ytest, ytest_pred_score, thres_opt)
    return metrics_te.merge(metrics_tr, on='thresh')


def tr_predict(df_new, col_y, target_names=['0', '1'], model='LR', penalty='l1',
               cv_folds=5, scoring='f1', test_size=0.2, report=False, RFE=False, pred_score=False):
    # scaler = preprocessing.StandardScaler()#MinMaxScaler
    y = df_new[col_y].values
    metrics_all = []

    if is_BWH:
        my_seeds = range(2020, 2021)
    else:
        my_seeds = range(2040, 2045)

    for seed in my_seeds:
        X = df_new.drop([col_y], axis=1).values
        name_cols = df_new.drop([col_y], axis=1).columns.values

        if is_BWH:
            X = pd.DataFrame(X)
            y = pd.DataFrame(y)

            X_train = X.loc[Train_Index, :]
            xtest = X.loc[Test_Index, :]
            y_train = y.loc[Train_Index, :]
            ytest = y.loc[Test_Index, :]

        else:
            X_train, xtest, y_train, ytest = train_test_split(
                X, y, stratify=y, test_size=test_size, random_state=seed)

        if RFE:
            df_train = pd.DataFrame(X_train, columns=name_cols)
            df_train[col_y] = y_train
            # my_penalty='l1', my_C = 1, my_range=range(25,46,5),
            id_keep_1st, F1_all_rfe = my_RFE(
                df_train, col_y=col_y, cvFolds=cv_folds, scoring=scoring)
            print(F1_all_rfe)
            X_train = df_train[id_keep_1st]
            df_test = pd.DataFrame(xtest, columns=name_cols)
            xtest = df_test[id_keep_1st]
            name_cols = id_keep_1st

        clf = my_train(
            X_train,
            y_train,
            model=model,
            penalty=penalty,
            cv=cv_folds,
            scoring=scoring,
            class_weight='balanced',
            seed=seed)
        metrics_all.append(
            my_test(
                X_train,
                xtest,
                y_train,
                ytest,
                clf,
                target_names,
                report=report,
                model=model))
    metrics_df = pd.concat(metrics_all)
    metrics_df = metrics_df[cols_rep].describe(
    ).T[['mean', 'std']].stack().to_frame().T
    # refit using all samples to get non-biased coef.
    clf.fit(X, y)
    if pred_score:
        if model == 'SVM':
            y_pred_score = clf.decision_function(X)
        else:
            y_pred_score = clf.predict_proba(X)[:, 1]
        df_new['y_pred_score'] = y_pred_score
    if model == 'LGB' or model == 'RF':
        df_coef_ = pd.DataFrame(list(zip(name_cols, np.round(
            clf.feature_importances_, 2))), columns=['Variable', 'coef_'])
    else:
        df_coef_ = pd.DataFrame(
            list(zip(name_cols, np.round(clf.coef_[0], 2))), columns=['Variable', 'coef_'])
        df_coef_ = df_coef_.append({'Variable': 'intercept_', 'coef_': np.round(
            clf.intercept_, 2)}, ignore_index=True)
    df_coef_['coef_abs'] = df_coef_['coef_'].abs()
    if pred_score:  # ==True
        return df_coef_.sort_values('coef_abs', ascending=False)[
            ['Variable', 'coef_']], metrics_df, df_new['y_pred_score']  # , scaler
    else:
        return df_coef_.sort_values('coef_abs', ascending=False)[
            ['Variable', 'coef_']], metrics_df


# ------------------------------------------------------------------------------
#    Hospitalization Models
# ------------------------------------------------------------------------------


# Test split

is_BWH = 1     # 1 for BWH as test and 0 for random split

cols_rep = [
    'AUC',
    'micro F1-score',
    'weighted F1-score',
    'tr AUC',
    'tr micro F1-score',
    'tr weighted F1-score']


# Load hospital names

HospitalNames = pd.read_csv('hos_stat_latest.csv', index_col=0, header=None)
HospitalNames["PID"] = HospitalNames.index
HospitalNames = HospitalNames.reset_index(drop=True)
HospitalNames = HospitalNames.sort_values(by=['PID'])
HospitalNames = HospitalNames.reset_index(drop=True)


Train_Index = list(HospitalNames[HospitalNames[1] != 'BWH'].index)
Test_Index = list(HospitalNames[HospitalNames[1] == 'BWH'].index)


df = X


# Remove high correlated
print(high_corr(df, thres=0.8))
df = df.drop(columns=['Alcohol_No'])


# All features -----------------------------------------------------------
Data1_DF_All = pd.concat([Y, df], axis=1)

# Statistical feature selection

result = stat_test(df, Y['Hospitalization'])
drop_cols = result.loc[result['p-value'] > 0.05, 'Variable'].values
df_new = df_drop(df, drop_cols)
df_new.shape
Data1_DF22 = pd.concat([Y, df_new], axis=1)

# Select a model ['LR','SVM','RF','LGB'] and a penalty [l1',l2']

df_coef_, metrics_df = tr_predict(
    Data1_DF_All, col_y='Hospitalization', target_names=[
        '0', '1'], model='LR', penalty='l1', cv_folds=5, scoring='roc_auc', test_size=0.2, report=True)
print(metrics_df.describe())


Ready_Table = metrics_df[['micro F1-score',
                          'AUC',
                          'weighted F1-score']].describe().T[['mean',
                                                              'std']].stack().to_frame().T
Ready_Table_All = (metrics_df.describe())


# Top Features   ---------------------------------------------------------

result = stat_test(df, Y['Hospitalization'])
drop_cols = result.loc[result['p-value'] > 0.05, 'Variable'].values
df_new = df_drop(df, drop_cols)
Selected_46_features = df_new.columns
my_C = 1
my_penalty = 'l1'

names = df_new.columns
clf = LogisticRegression(
    C=my_C,
    penalty=my_penalty,
    class_weight='balanced',
    solver='liblinear')
rfe = RFE(clf, 11, step=1)
rfe.fit(df_new, Y['Hospitalization'])
id_keep_1st = list(names[rfe.ranking_ == 1].values)
print(id_keep_1st)


# Cumpute Odds Ratios   --------------------------------------------------
# Load binarized data
Y_Hyper = pd.read_csv('Final_Y_binarized.csv')
X_Hyper = pd.read_csv('Final_X_binarized.csv')

logit = sm.Logit(Y_Hyper)
result_Hyper = logit.fit()

# odds ratios and 95% CI + Coef
params = result_Hyper.params
conf = result_Hyper.conf_int()
conf['OR'] = params
conf.columns = ['Odds_Ratio 2.5%', 'Odds_Ratio 97.5%', 'Odds_Ratio']
Coef_CI = pd.concat([result_Hyper.conf_int(), result_Hyper.params], axis=1)
Coef_CI.columns = ['Coef_Binary 2.5%', 'Coef_Binary 97.5%', 'Coef_Binary']
Oods_Ratio_CI = np.exp(conf)
print(Oods_Ratio_CI)

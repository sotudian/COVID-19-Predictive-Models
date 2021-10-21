# %  ---------------------------------------------------------------------------------------------------------------------------------------
# %  Intubation Models
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



from sklearn.feature_selection import RFE  # ,RFECV #
import lightgbm as lgb
from sklearn.metrics import classification_report, f1_score, roc_curve, auc, accuracy_score
from warnings import filterwarnings
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from scipy import stats
from sklearn import preprocessing
import pandas as pd
import numpy as np





# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------


pd.options.display.max_columns = None
pd.options.display.max_rows = None


def chi2_cols(y, x):
    y_list = y.astype(int).tolist()
    x_list = x.astype(int).tolist()
    freq = np.zeros([2, 2])

    for i in range(len(y_list)):
        if y_list[i] == 0 and x_list[i] == 0:
            freq[0, 0] += 1
        if y_list[i] == 1 and x_list[i] == 0:
            freq[1, 0] += 1
        if y_list[i] == 0 and x_list[i] == 1:
            freq[0, 1] += 1
        if y_list[i] == 1 and x_list[i] == 1:
            freq[1, 1] += 1
    y_0_sum = np.sum(freq[0, :])
    y_1_sum = np.sum(freq[1, :])
    x_0_sum = np.sum(freq[:, 0])
    x_1_sum = np.sum(freq[:, 1])
    total = y_0_sum + y_1_sum
    y_0_ratio = y_0_sum / total
    freq_ = np.zeros([2, 2])
    freq_[0, 0] = x_0_sum * y_0_ratio
    freq_[0, 1] = x_1_sum * y_0_ratio
    freq_[1, 0] = x_0_sum - freq_[0, 0]
    freq_[1, 1] = x_1_sum - freq_[0, 1]

    stat, p_value = stats.chisquare(freq, freq_, axis=None)
    return p_value  # stat,


def stat_test(df, y):
    name = pd.DataFrame(df.columns, columns=['Variable'])
    df0 = df[y == 0]
    df1 = df[y == 1]
    pvalue = []
    y_corr = []
    for col in df.columns:
        if df[col].nunique() == 2:
            pvalue.append(chi2_cols(y, df[col]))
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
        df[col] = df[col].fillna(df[col].median())
    return df


def df_drop(df_new, drop_cols):
    return df_new.drop(df_new.columns[df_new.columns.isin(drop_cols)], axis=1)


def clf_F1(best_C_grid, best_F1, best_F1std, classifier, X_train,
           y_train, C_grid, nFolds, silent=True, seed=2020, scoring='f1'):
    results = cross_val_score(
        classifier,
        X_train,
        y_train,
        cv=StratifiedKFold(
            n_splits=nFolds,
            shuffle=True,
            random_state=seed),
        n_jobs=-1,
        scoring=scoring)  # cross_validation.
    F1, F1std = results.mean(), results.std()
    if silent == False:
        print(C_grid, F1, F1std)
    if F1 > best_F1:
        best_C_grid = C_grid
        best_F1, best_F1std = F1, F1std
    return best_C_grid, best_F1, best_F1std


def my_RFE(df_new, col_y='flag_ICU', my_range=range(20, 36, 1),
           my_penalty='l1', my_C=0.1, cvFolds=5, step=1, scoring='f1'):
    F1_all_rfe = []
    Xraw = df_new.drop(col_y, axis=1).values
    y = df_new[col_y].values
    names = df_new.drop(col_y, axis=1).columns
    for n_select in my_range:
        scaler = preprocessing.StandardScaler()  # MinMaxScaler
        X = scaler.fit_transform(Xraw)
        clf = LogisticRegression(
            C=my_C,
            penalty=my_penalty,
            class_weight='balanced',
            solver='liblinear')  # tol=0.01,
        # clf = LinearSVC(penalty='l1',C=0.1,class_weight= 'balanced', dual=False)
        rfe = RFE(clf, n_select, step=step)
        rfe.fit(X, y.ravel())
        X = df_new.drop(col_y, axis=1).drop(
            names[rfe.ranking_ > 1], axis=1).values
        X = scaler.fit_transform(X)
        best_F1, best_F1std = 0.1, 0
        best_C_grid = 0
        for C_grid in [0.01, 0.1, 1, 10]:
            clf = LogisticRegression(
                C=C_grid,
                class_weight='balanced',
                solver='liblinear')  # penalty=my_penalty,
            best_C_grid, best_F1, best_F1std = clf_F1(
                best_C_grid, best_F1, best_F1std, clf, X, y, C_grid, cvFolds, scoring=scoring)
        F1_all_rfe.append((n_select, best_F1, best_F1std))
    F1_all_rfe = pd.DataFrame(
        F1_all_rfe,
        index=my_range,
        columns=[
            'n_select',
            "best_score_mean",
            "best_score_std"])
    # -F1_all_rfe['best_score_std']
    F1_all_rfe['best_score_'] = F1_all_rfe['best_score_mean']
    ######
    ######
    X = scaler.fit_transform(Xraw)
    clf = LogisticRegression(
        C=my_C,
        penalty=my_penalty,
        class_weight='balanced',
        solver='liblinear')  # 0.
    rfe = RFE(
        clf, F1_all_rfe.loc[F1_all_rfe['best_score_'].idxmax(), 'n_select'], step=step)
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
            max_iter=5000)  # , tol=0.0001
        # 'kernel':('linear', 'rbf'),
        param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
        gsearch = GridSearchCV(svc, param_grid, cv=cv, scoring=scoring)
    elif model == 'LGB':
        param_grid = {
            'num_leaves': range(2, 7, 1),
            'n_estimators': range(40, 100, 10),
            # [0.6, 0.7, 0.8, 0.9]#[0.6, 0.75, 0.9]
            'colsample_bytree': [0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
            # 'reg_alpha': [0.1, 0.5],# 'min_data_in_leaf': [30, 50, 100, 300, 400],
            # 'lambda_l1': [0, 1, 1.5],# 'lambda_l2': [0, 1]
        }
        lgb_estimator = lgb.LGBMClassifier(
            boosting_type='gbdt',
            objective='binary',
            learning_rate=0.1,
            class_weight=class_weight,
            random_state=seed)  # eval_metric='auc' num_boost_round=2000,
        gsearch = GridSearchCV(
            estimator=lgb_estimator,
            param_grid=param_grid,
            cv=cv,
            n_jobs=-1,
            scoring=scoring)
    elif model == 'RF':
        rfc = RandomForestClassifier(
            n_estimators=100,
            random_state=seed,
            class_weight=class_weight,
            n_jobs=-1)
        param_grid = {
            # , 0.4, 0.5, 0.6, 0.7, 0.8 [ 'sqrt', 'log2',15],#'auto'  1.0/3,
            'max_features': [0.05, 0.1, 0.2, 0.3],
            'max_depth': range(2, 6, 1)  # [2, 10]
            #     'criterion' :['gini', 'entropy'] #min_samples_split = 10,
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
        parameters = {'C': [0.1, 1, 10]}
        gsearch = GridSearchCV(LR, parameters, cv=cv, scoring=scoring)
        # clf = LogisticRegressionCV(Cs=[10**-1,10**0, 10], penalty=penalty,
        # class_weight= class_weight,solver='liblinear', cv=cv,
        # scoring=scoring, random_state=seed)#, tol=0.01
    gsearch.fit(X_train, y_train)
    clf = gsearch.best_estimator_
    if model == 'LGB' or model == 'RF':
        print('Best parameters found by grid search are:', gsearch.best_params_)
    # print('train set accuracy:', clf.score(X_train, y_train))
    # print('train set accuracy:', gsearch.best_score_)
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
    # fpr, tpr, thresholds = roc_curve(ytest, ytest_pred_score)
    # if report:
    #     print(classification_report(ytest, ytest_pred, target_names=target_names,digits=4))
    metrics_te = cal_f1_scores_te(ytest, ytest_pred_score, thres_opt)
    return metrics_te.merge(
        metrics_tr, on='thresh'), thres_opt, ytest_pred_score


def tr_predict(df_new, col_y, target_names=['0', '1'], model='LR', penalty='l1',
               cv_folds=5, scoring='f1', test_size=0.2, report=False, RFE=False, pred_score=False):
    scaler = preprocessing.StandardScaler()  # MinMaxScaler
    y = df_new[col_y].values
    metrics_all = []
    if is_BWH:
        my_seeds = range(2020, 2021)
        # DATA_PATH = '/content/gdrive/Shared drives/Covid/Finalized_Processed_Dataset/'
        # hos_stat_latest = pd.read_csv(DATA_PATH +
        # 'Dataset_v0.4/hos_stat_latest.csv')#,index_col=0
        mask = df_new.index.isin(hos_stat_latest.loc[(
            hos_stat_latest['Hospital'] != 'BWH'), 'PID'].values)
    else:
        my_seeds = range(2020, 2025)
    for seed in my_seeds:
        X = df_new.drop([col_y], axis=1).values
        name_cols = df_new.drop([col_y], axis=1).columns.values
        X = scaler.fit_transform(X)
        if is_BWH:
            X_train, xtest, y_train, ytest = X[mask,
                                               :], X[~mask, :], y[mask], y[~mask]
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
        metrics_te, thres_opt, ytest_pred_score = my_test(
            X_train, xtest, y_train, ytest, clf, target_names, report=report, model=model)
        metrics_all.append(metrics_te)
    metrics_df = pd.concat(metrics_all)
    metrics_df = metrics_df[cols_rep].describe(
    ).T[['mean', 'std']].stack().to_frame().T

    if pred_score and is_BWH:
        ytest_pred = df_new.loc[~mask, [col_y]].copy()
        # if model=='SVM':
        #     y_pred_score=clf.decision_function(xtest)
        # else:
        #     y_pred_score=clf.predict_proba(xtest)[:,1]
        # print(ytest_pred_score)
        print(ytest_pred.shape, ytest_pred.head())
        print('thres_opt', thres_opt)
        ytest_pred['ytest_pred_score'] = ytest_pred_score
        ytest_pred['ytest_pred'] = ytest_pred_score > thres_opt
        ytest_pred['ytest'] = ytest
        # refit using all samples to get non-biased coef.
    clf.fit(X, y)
    if model == 'LGB' or model == 'RF':
        df_coef_ = pd.DataFrame(list(zip(name_cols, np.round(
            clf.feature_importances_, 2))), columns=['Variable', 'coef_'])
    else:
        df_coef_ = pd.DataFrame(
            list(zip(name_cols, np.round(clf.coef_[0], 2))), columns=['Variable', 'coef_'])
        df_coef_ = df_coef_.append({'Variable': 'intercept_', 'coef_': np.round(
            clf.intercept_, 2)}, ignore_index=True)
    df_coef_['coef_abs'] = df_coef_['coef_'].abs()
    if pred_score and is_BWH:  # ==True
        return df_coef_.sort_values('coef_abs', ascending=False)[
            ['Variable', 'coef_']], metrics_df, ytest_pred
    else:
        return df_coef_.sort_values('coef_abs', ascending=False)[
            ['Variable', 'coef_']], metrics_df


def get_odds_ratio(df, col_y='flag_ventilator'):

    import statsmodels.api as sm
    import pylab as pl

    X = df.drop(columns=[col_y])
    Y = df[col_y]

    # logit = sm.Logit(Y, X)
    logit = sm.Logit(Y, sm.add_constant(X))

    # fit the model
    result = logit.fit_regularized()
    print(result.summary())

    # odds ratios and 95% CI + Coef
    Coef_CI = pd.concat([result.params,
                         np.exp(result.params),
                         result.pvalues,
                         np.exp(result.conf_int()).astype(float),
                         result.conf_int()],
                        axis=1)
    Coef_CI.columns = [
        'Coef_Binary',
        'Odds_Ratio',
        'P_Value_Coef',
        'Odds_Ratio_2.5%',
        'Odds_Ratio_97.5%',
        'Coef_Binary_2.5%',
        'Coef_Binary_97.5%']
    print(Coef_CI)
    Coef_CI['Abs_Coef_Binary'] = Coef_CI['Coef_Binary'].abs()

    # .sort_values(['Abs_Coef_Binary'], ascending = False).drop(['Abs_Coef_Binary'], axis = 1) #= get_odds_ratio(df, col_y = 'flag_ventilator')
    return Coef_CI

def score_psi(df_psi, col_psi='psi', test_size=0.2):
    y = df_psi[col_y]  # df_new[col_y].values
    metrics_all = []
    if is_BWH:
        my_seeds = range(2020, 2021)
        # DATA_PATH = '/content/gdrive/Shared drives/Covid/Finalized_Processed_Dataset/'
        # hos_stat_latest = pd.read_csv(DATA_PATH +
        # 'Dataset_v0.4/hos_stat_latest.csv')#,index_col=0
        mask = df_psi['PID'].isin(hos_stat_latest.loc[(
            hos_stat_latest['Hospital'] != 'BWH'), 'PID'].values)
    else:
        my_seeds = range(2020, 2025)
    for seed in my_seeds:
        X = df_psi[col_psi].values
        if is_BWH:
            X_train, xtest, y_train, ytest = X[mask], X[~mask], y[mask], y[~mask]
        else:
            X_train, xtest, y_train, ytest = train_test_split(
                X, y, stratify=y, test_size=test_size, random_state=seed)

        metrics_tr = cal_f1_scores(y_train, X_train)
        thres_opt = metrics_tr['thresh'].values[0]
        print(thres_opt)
        metrics_te = cal_f1_scores_te(ytest, xtest, thres_opt)
        metrics_all.append(metrics_te.merge(metrics_tr, on='thresh'))
    metrics_df = pd.concat(metrics_all)
    return metrics_df[cols_rep].describe().T[['mean', 'std']
                                             ].stack().to_frame().T

def binarized(df):
        # 1 if LDH >=250, 0 otherwise
    # 1 if CRP >=10, 0 otherwise
    # 1 if Anion gap >=12, 0 otherwise
    # 1 if Glucose >=110, 0 otherwise
    # 1 if Pulse >= 100 for Pulse, 0 otherwise
    # 1 if Respiratory_Rate >= 20, 0 otherwise
    # 1 if age>= 65, 0 otherwise
    # Globulin >=4
    # 1 Globulin <=2 or >=4, 0 otherwise
    # For vitals:
    # 1 if BMI>=30, 0 otherwise
    # 1 if Temp>=37.5 Celsius or 98.7 Farenheit, 0 otherwise

    feas_lb_thresh = [
        'LDH',
        'CRP (mg/L)',
        'Anion Gap',
        'Glucose',
        'Pulse',
        'Respiratory_Rate',
        'Age',
        'Globulin',
        'BMI',
        'Temperature']
    thresh_lb = [250, 10, 12, 110, 100, 20, 65, 4, 30, 37.5]

    newfeas_lb_thresh = []
    feas2compare_lb_thresh = []
    for i, fea in enumerate(feas_lb_thresh):
        if fea in df.columns:
            # .str.extract("([-+]?\d*\.\d+|[-+]?\d+)")
            df[fea] = df[fea].astype(float)
            new_fea = fea + '_gt_' + str(thresh_lb[i])
            df[new_fea] = (df[fea] >= thresh_lb[i]).astype(int)
            newfeas_lb_thresh.append(new_fea)
            feas2compare_lb_thresh += [fea, new_fea]
    # 1 if Calcium <=8.5, 0 otherwise
    # total_protein <=6.5
    # 1 if Total protein <=6.5 or >=8.3, 0 otherwise
    # 1 if GFR <=60, 0 otherwise
    # 1 if Sodium <135, 0 otherwise
    # For vitals:
    # 1 if SpO2  <= 94, 0 otherwise
    # 1 if SysBP <= 100, 0 otherwise
    # 1 if Dia_BP <= 60, 0 otherwise

    # # feas_ub_thresh = ['Calcium', 'Total Protein', 'GFR (estimated)', 'Sodium', 'SpO2_percentage', 'Systolic_BP', 'Diastolic_BP']
    # # thresh_ub = [8.5, 6.5, 60, 135, 100, 94, 100, 60]
    feas_ub_thresh = [
        'Calcium',
        'Total Protein',
        'GFR (estimated)',
        'Sodium',
        'SpO2_percentage',
        'Systolic_BP',
        'Diastolic_BP',
        'Albumin',
        'Chloride']
    thresh_ub = [8.5, 6.5, 60, 135, 94, 100, 60, 3.3, 95]

    newfeas_ub_thresh = []
    feas2compare_ub_thresh = []
    for i, fea in enumerate(feas_ub_thresh):
        if fea in df.columns:
            # str.extract("([-+]?\d*\.\d+|[-+]?\d+)").
            df[fea] = df[fea].astype(float)
            new_fea = fea + '_lt_' + str(thresh_ub[i])
            df[new_fea] = (df[fea] <= thresh_ub[i]).astype(int)
            newfeas_ub_thresh.append(new_fea)
            feas2compare_ub_thresh += [fea, new_fea]

    print(df[feas2compare_lb_thresh + feas2compare_ub_thresh].head(1))

    df = df_drop(df, feas_lb_thresh)
    df = df_drop(df, feas_ub_thresh)
    return df, feas2compare_lb_thresh + feas2compare_ub_thresh





# -----------------------------------------------------------------------------
# ICU Models
# -----------------------------------------------------------------------------

# Parameters

is_BWH = True   # True for BWH as test and False for random split
col_y = 'flag_ventilator' 
cols_rep = [
    'AUC',
    'micro F1-score',
    'weighted F1-score',
    'tr AUC',
    'tr micro F1-score',
    'tr weighted F1-score']



# -----------------------------------------------------------------------------
# Data pre-processing
# -----------------------------------------------------------------------------

df = pd.read_csv('Final_Data_Medications_Comorbidity_Symptom_Vitals_radiology.csv')
df.loc[df['PID'].isin([182, 603, 606, 621, 1112]), col_y] = 1
df2 = pd.read_csv('Final_Data_LabTests.csv')
df_PID = pd.read_csv('Final_Data_Acceptable_missing.csv')
df_names = pd.read_csv('Final_Data_Labeles.csv')

df2.columns = df_names['Name'].values
df2 = df2[df2['PID'].isin(df_PID['PID'])]
print(df.shape, df2.shape)
print(df.columns, df2.columns)

df2_nan = pd.read_csv('Finall_Dataset_LabTests_AllPatients_BeforeIMPUTATION.csv',
    index_col=0)
df2_nan = df2_nan[df2_nan['PID'].isin(df_PID['PID'])]
df2_nan.columns = df_names['Name'].values
print(df2_nan.shape)

df2_nan_ = df2_nan.dropna(
    axis=0,
    subset=[
        'HGB',
        'RDW',
        'PLT',
        'MCH',
        'MCHC',
        'NRBC% (auto)',
        'MCV',
        'RBC',
        'WBC',
        'MPV',
        'HCT'])
print(df2_nan_.shape)
df2 = df2[df2['PID'].isin(df2_nan_['PID'])]
print(df2.shape)
df2_nan_ = df2_nan_[df2_nan_['PID'].isin(df['PID'])]
df2_nan_.shape

hos_stat_latest = pd.read_csv(
    DATA_PATH +
    'Dataset_v0.4/hos_stat_latest.csv')  # ,index_col=0
hos_stat_latest.head()

hos_stat_latest.shape

df = df.merge(
    df2.drop(['Gender', 'Race', 'Age', 'Language', ], axis=1), on='PID')
print(df.shape)

"""## select patients with non-missing LDH and CRP (mg/L) so almost no missing for top 10.
84%-85% AUC for 669 patients.
"""

df3 = df2_nan_.copy().reset_index()
df3 = df_drop(df3, ['Language', 'index'])
cat_col = ['Gender', 'Race', 'Tobacco', 'Alcohol']  # 'Marital_status',
# # ['LDH', 'CRP (mg/L)','Lactic acid (mmol/L)','D-Dimer (ng/mL)']:#
df3 = pd.get_dummies(df3, prefix=cat_col, columns=cat_col, drop_first=True)
# ['LDH', 'CRP (mg/L)','Lactic acid (mmol/L)','D-Dimer (ng/mL)']:#
for col in df3.select_dtypes(include=['object']).columns:
    # print(col)
    df3[col] = df3[col].str.extract(r"([-+]?\d*\.\d+|[-+]?\d+)").astype(float)

df3_impute = df3.copy()

print(df3[['LDH', 'CRP (mg/L)', 'Lactic acid (mmol/L)']].describe())
mask = df3[['LDH', 'CRP (mg/L)']].notnull().all(axis=1)
# mask.sum()
df = df[mask]
print(df.shape)

# df3.describe()
df['PID'].head()

df = df.merge(hos_stat_latest, on='PID')
df['Hospital'].value_counts(), df.groupby(['Hospital'])[col_y].mean()

# df['is_BWH']=(df['Hospital']=='BWH')

"""## cols_impute"""


# ,'Albumin','Eos#','"Granulocytes, immature"'
cols_impute = ['D-Dimer (ng/mL)']
for col_ in cols_impute:
    mask1 = df3[col_].notnull()
    cols_ = df3[~mask].columns

    print('non-missing # ', mask1.sum())
    X_train = df3[cols_].loc[mask1, :]
    xtest = df3[cols_].loc[~mask1, :]
    y_train = df3[col_].loc[mask1]
    # # Create our imputer to replace missing values with the mean e.g.
    imp = SimpleImputer(
        missing_values=np.nan,
        strategy='constant',
        fill_value=-99)  # 'median'
    X_train = imp.fit_transform(X_train)
    xtest = imp.transform(xtest)
    param_grid = {
        # ,40,5,6,7,8, [0.05, 0.1,0.2],#  [0.1,0.2,0.3, 0.4],# , 0.5
        'max_features': [20, 30],
        'max_depth': [5, 10, 15, 20]  # 3,,6  3, 4,5,
    }  # 'min_samples_split': [5] # [ 'sqrt', 'log2',1.0/3],#'auto'  1.0/3,
    regr = RandomForestRegressor(
        random_state=2020,
        n_estimators=100,
        n_jobs=-
        1)  # , oob_score=True criterion='mae',
    gsearch = GridSearchCV(
        estimator=regr,
        param_grid=param_grid,
        cv=5,
        scoring='r2')  # , return_train_score=True
    gsearch.fit(X_train, y_train)
    print('Best parameters found by grid search are:', gsearch.best_params_)
    print('train set score:', gsearch.best_score_)
    clf = gsearch.best_estimator_
    df3_impute.loc[~mask1, col_] = np.round(clf.predict(xtest), 2)


"""## AUC improved for single var."""


for col_ in cols_impute:
    temp = df[col_]
    fpr, tpr, thresholds = roc_curve(df[col_y], temp)
    print(temp.corr(df[col_y]), auc(fpr, tpr))

print('after imputation:')
for col_ in cols_impute:
    temp = df3_impute[mask][col_]
    fpr, tpr, thresholds = roc_curve(df[col_y], temp)
    print(temp.corr(df[col_y]), auc(fpr, tpr))


for col_ in cols_impute:
    df[col_] = df3_impute[mask][col_].values


df_vent_date = pd.read_csv('Final_dates_ICU_Ventilator_new.csv')
if col_y == 'flag_ventilator':
    PIDs2drop = df_vent_date.loc[df_vent_date['Hours_fromLABtoVentilator'] < 4, 'PID']
if col_y == 'flag_ICU':
    PIDs2drop = df_vent_date.loc[df_vent_date['Hours_fromLABtoICU'] < 4, 'PID']

df = df[~df['PID'].isin(PIDs2drop)]
df3 = df3[~df3['PID'].isin(PIDs2drop)]
df2_nan_ = df2_nan_[~df2_nan_['PID'].isin(PIDs2drop)]
print(df.shape, df3.shape, df2_nan_.shape)


"""# psi/curb_65"""
psi_imputed = pd.read_csv('psi_imputed.csv')
df_psi = psi_imputed[['PID', 'psi']].merge(df[['PID', col_y]])

curb_65_imputed = pd.read_csv('curb_65_imputed.csv')
df_curb_65 = curb_65_imputed[['PID', 'curb_65']].merge(df[['PID', col_y]])
df_psi.head()




df_re_psi = score_psi(df_psi, col_psi='psi', test_size=0.2)
df_re_curb_65 = score_psi(df_curb_65, col_psi='curb_65', test_size=0.2)

df.head(1)

"""1172-->1050 since 121 samples have no main lab tests"""

# df_count.sort_values().index
df[col_y].mean(), df[col_y].sum(), df['LDH'].corr(df[col_y])

df.describe()


"""# preprocess

## drop as required
"""

df.index = df['PID']

drop_cols = ['Iron', 'TIBC', 'Osmolality', '"Base Deficit, arterial"', 'Myelocytes',
             'Plasma cells (%)', 'Metamyelocytes', '"Calcium, ionized (mmol/L)"',
             '"pH, Arterial"', 'Bands (manual)', '"Base Deficit, venous"',
             '"PCO2, Arterial"', '"PO2, Arterial"', 'Height',
             '"HCO3, unspecified "', 'Phosphorus', 'Weight', 'Lipase (U/L)',
             '"PO2, Venous"', '"pH, Venous"', '"PCO2, Venous"', 'Carbon Dioxide', 'Magnesium', 'symptom_Nausea', 'Temperature']

df = df_drop(df, drop_cols)
print(df.shape)

"""## high_corr
"""

print(high_corr(df, thres=0.8))
drop_cols = ['PID', 'Hospital', 'Language', 'Osmolality',
             'MCH',
             'HCT', 'RBC', '"NRBC#, auto"', 'ALT', 'AST', 'Ferritin', 'CK',
             'Iron', 'TIBC',
             'Lymph#', 'Chloride', 'Total Protein', 'GFR (estimated)'
             ]
if col_y == 'flag_ICU':
    drop_cols += ['flag_ventilator']
if col_y == 'flag_ventilator':
    drop_cols += ['flag_ICU']
df = df_drop(df, drop_cols)
print(high_corr(df, thres=0.7))

df.dtypes.unique()  # df_new.dtypes.unique()

"""## df_new"""

cat_col = ['Gender', 'Race', 'Marital_status', 'Tobacco', 'Alcohol']
df_new = pd.get_dummies(
    df,
    prefix=cat_col,
    columns=cat_col,
    drop_first=True)  # , dummy_na=True
df_new = df_fillna(df_new)
df_new.shape

"""## df_new_std"""

# df_new.columns.values
# df_new.dtypes
df_new_std = df_new.std()
drop_cols = ['Language_English-ENGLISH', 'Alcohol_No',
             'Fibrinogen',
             'Marital_status_Unknown-@',
             'PID', 'COVID_Diag_Date', 'reference_date'] + list(df_new_std[df_new_std < 0.05].index)
df_new = df_drop(df_new, drop_cols)
print(high_corr(df_new, thres=0.8))
df_new.shape

print(high_corr(df_new, thres=0.7))

result = stat_test(df_new, df_new[col_y])
result

df_new.shape

"""# AUC or F1"""

my_scoring = 'roc_auc'

cols = [
    'LDH',
    'Calcium',
    'Anion Gap',
    'CRP (mg/L)',
    'Glucose',
    'Neutrophil #',
    'Sodium',
    'Albumin'] 
df[cols].describe()

threshold = .99
cols = [
    'LDH',
    'Calcium',
    'Anion Gap',
    'CRP (mg/L)',
    'Glucose',
    'Neutrophil #'] 
for col in cols:
    mask_select = df_new[col] > df_new[col].quantile([threshold]).values[0]
    df_new.loc[mask_select, col] = df_new[col].quantile([threshold]).values[0]
    mask_select = df_new[col] < df_new[col].quantile([1 - threshold]).values[0]
    df_new.loc[mask_select, col] = df_new[col].quantile(
        [1 - threshold]).values[0]

# df_new[cols].describe()
mask = df3[['LDH', 'CRP (mg/L)']].notnull().all(axis=1)
df_count = pd.DataFrame(
    df3[mask].count().sort_values(
        ascending=False),
    columns=['count'])  # .reset_index()
# .to_frame( )#col=
df_count['Variable'] = df_count.index
# df_count.head()

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


filterwarnings('ignore')

df_coef_, metrics_df = tr_predict(
    df_new, col_y=col_y, target_names=[
        '0', '1'], model='LR', penalty='l1', cv_folds=5, scoring=my_scoring)
# df_AUCs=pd.concat([df_AUCs,df_AUC.rename(index={0: 'LR-L1'})])
df_AUCs = metrics_df.rename(index={0: 'LR-L1'})
# # df_coef_[df_coef_['coef_']!=0]

df_coef_, metrics_df = tr_predict(
    df_new, col_y=col_y, target_names=[
        '0', '1'], model='SVM', penalty='l1', cv_folds=5, scoring=my_scoring)
# metrics_df.describe()
df_AUCs = pd.concat([df_AUCs, metrics_df.rename(index={0: 'SVM-L1'})])

df_coef_, metrics_df = tr_predict(
    df_new, col_y=col_y, target_names=[
        '0', '1'], model='LGB', penalty='l1', cv_folds=5, scoring=my_scoring)
df_AUCs = pd.concat([df_AUCs, metrics_df.rename(index={0: 'GBT'})])
# df_coef_.merge(result, on='Variable').merge(df_count,how='left', on='Variable').fillna(len(df))[['Variable','coef_','y_corr','p-value','count','y1_mean', 'y0_mean']]

df_coef_, metrics_df = tr_predict(
    df_new, col_y=col_y, target_names=[
        '0', '1'], model='RF', penalty='l1', cv_folds=5, scoring=my_scoring)
df_AUCs = pd.concat([df_AUCs, metrics_df.rename(index={0: 'RF'})])
# df_coef_.merge(result, on='Variable').merge(df_count,how='left', on='Variable').fillna(len(df))[['Variable','coef_','y_corr','p-value','count','y1_mean', 'y0_mean']]
df_AUCs

"""# after stat select"""

drop_cols = result.loc[result['p-value'] > 0.05, 'Variable'].values
# drop_cols=result.loc[result['p-value']>0.1,'Variable'].values
df_new = df_drop(df_new, drop_cols)
df_new.shape

df_new.columns

"""## drop less frequent"""

# df_new.describe()
drop_cols = []
for col in df_new.columns:
    if (df_new[col].nunique() == 2) & (df_new[col].std() < 0.2):
        drop_cols.append(col)
        # print(col, df_new[col].std())
df_new = df_drop(df_new, drop_cols)
df_new.shape

df_coef_, metrics_df = tr_predict(
    df_new, col_y=col_y, target_names=[
        '0', '1'], model='LR', penalty='l1', cv_folds=5, scoring=my_scoring)
df_AUCs = pd.concat([df_AUCs, metrics_df.rename(index={0: 'LR-L1_'})])

df_coef_, metrics_df = tr_predict(
    df_new, col_y=col_y, target_names=[
        '0', '1'], model='SVM', penalty='l1', cv_folds=5, scoring=my_scoring)
df_AUCs = pd.concat([df_AUCs, metrics_df.rename(index={0: 'SVM-L1_'})])

df_coef_, metrics_df = tr_predict(
    df_new, col_y=col_y, target_names=[
        '0', '1'], model='LGB', penalty='l1', cv_folds=5, scoring=my_scoring)
df_AUCs = pd.concat([df_AUCs, metrics_df.rename(index={0: 'GBT_'})])

df_coef_, metrics_df = tr_predict(
    df_new, col_y=col_y, target_names=[
        '0', '1'], model='RF', penalty='l1', cv_folds=5, scoring=my_scoring)
df_AUCs = pd.concat([df_AUCs, metrics_df.rename(index={0: 'RF_'})])
df_AUCs


# -----------------------------------------------------------------------------
# TOP 10 + L1LR
# -----------------------------------------------------------------------------

my_penalty = 'l1'

Xraw = df_new.drop(col_y, axis=1).values
y = df_new[col_y].values
names = df_new.drop(col_y, axis=1).columns
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(Xraw)

AUC_best = 0.5
for my_C in [0.1, 1, 10]:
    clf = LogisticRegression(
        C=my_C,
        penalty=my_penalty,
        class_weight='balanced',
        solver='liblinear')  # 0.
    for n_select in range(1, 11):
        rfe = RFE(clf, n_select, step=1)
        rfe.fit(X, y.ravel())
        id_keep_1st = list(names[rfe.ranking_ == 1].values)
        # # print(id_keep_1st)
        df_coef_, metrics_df = tr_predict(df_new[id_keep_1st + [col_y]], col_y=col_y, target_names=[
                                          '0', '1'], model='LR', penalty='l1', cv_folds=5, scoring=my_scoring, report=False)  # test_size=0.2,
        # # metrics_df.describe()
        # # print(n_select)
        AUC_ = metrics_df['AUC']['mean'].values
        if AUC_ > AUC_best:
            AUC_best = AUC_
            id_keep_1st_best = id_keep_1st
            print(n_select, AUC_, id_keep_1st_best)

df_coef_, metrics_df, df_pred_score = tr_predict(df_new[id_keep_1st_best + [col_y]], col_y=col_y, target_names=[
                                                 '0', '1'], model='LR', penalty='l1', cv_folds=5, scoring=my_scoring, pred_score=True)  # test_size=0.2,, report=True
# metrics_df.describe()
if 'df_AUCs' in globals():
    df_AUCs = pd.concat([df_AUCs, metrics_df.rename(index={0: 'LR-L1-top'})])
else:
    df_AUCs = metrics_df.rename(index={0: 'LR-L1-top'})

result10 = df_coef_.merge(
    result,
    how='left',
    on='Variable').merge(
        df_count,
        how='left',
    on='Variable')

result10[['Variable', 'coef_', 'y_corr', 'p-value', 'y1_mean',
          'y0_mean', 'All_mean', 'All_std']]  # .fillna(len(df))



df_pred_score.head()

'pred_prob_test_' + col_y[5:] + '_' + str(is_BWH) + 'BWH.csv'

df_pred_score.to_csv('/Covid/Finalized_Processed_Dataset/codes_ICU/pred_prob_test_' +
                     col_y[5:] + '_' + str(is_BWH) + 'BWH.csv')



# -----------------------------------------------------------------------------
# Binary LR top 10
# -----------------------------------------------------------------------------


df_new2, feas2compare = binarized(df_new)


top10_list = list(result10['Variable'].values)  # id_keep_1st
top10_list.remove('intercept_')
top10_list

binary_l = []
for col1 in top10_list:
    is_in = False
    for col2 in feas2compare:
        if (col1 in col2) and (col1 != col2):
            binary_l.append(col2)
            is_in = True
    if is_in == False:
        binary_l.append(col1)
binary_l

my_scoring = 'roc_auc'
df_coef_, metrics_df = tr_predict(df_new[binary_l + [col_y]], col_y=col_y, target_names=[
                                  '0', '1'], model='LR', penalty='l1', cv_folds=5, scoring=my_scoring)
df_AUCs = pd.concat(
    [df_AUCs, metrics_df.rename(index={0: 'LR-L1-top-binary'})])
# df_AUCs.loc[['RF','GBT','SVM-L1','LR-L1','LR-L1-top','LR-L1-top-binary']]
# cols=binary_l

cols7 = ['medication_CCBs', 'comorbidity_Hypertension',
         'comorbidity_Arrhythmia', 'comorbidity_CHF', 'comorbidity_Diabetes',
         'medication_Beta-Blockers', 'medication_Diuretics']
# # df_new2[cols3]=df[cols3]

Coef_CI = get_odds_ratio(df_new[binary_l + [col_y]], col_y=col_y)  # +cols7
Coef_CI.reset_index(inplace=True)
Coef_CI.columns = ['Variable'] + Coef_CI.columns[1:].tolist()

Coef_CI.columns

# Coef_CI.columns
mask = ~Coef_CI['Variable'].isin(cols7)
Coef_CI.loc[mask, ['Variable', 'Coef_Binary', 'Odds_Ratio',
                   'Coef_Binary_2.5%', 'Coef_Binary_97.5%']]


# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------


intercept_ = result10.loc[result10['Variable']
                          == 'intercept_', 'coef_'].values[0]
intercept_
result10 = result10[result10['Variable']
                    != 'intercept_'].reset_index(drop=True)
cols_report = [
    'Coef_Binary',
    'Odds_Ratio',
    'Odds_Ratio_2.5%',
    'Odds_Ratio_97.5%']

if 'const' in Coef_CI['Variable'].values:
    temp = Coef_CI[~Coef_CI['Variable'].isin(cols7)]
    temp = temp[~temp['Variable'].isin(['const'])].reset_index()
    result10[cols_report] = temp[cols_report]
else:
    result10[cols_report] = Coef_CI.loc[mask, cols_report]

result10 = result10.append(
    {'Variable': 'intercept_', 'coef_': intercept_}, ignore_index=True)
result10[['Variable', 'coef_', 'y1_mean', 'y0_mean', 'p-value',
          'y_corr'] + cols_report + ['All_mean', 'All_std']]  # .columns

result10

"""# df_AUCs"""

df_AUCs = pd.concat([df_AUCs, df_re_psi.rename(index={0: 'PSI'})])
df_AUCs = pd.concat([df_AUCs, df_re_curb_65.rename(index={0: 'curb_65'})])
# df_AUCs

# df_AUCs.round(4)['AUC']
df_AUCs_p = df_AUCs.round(3).applymap(
    lambda n: '{:.1%}'.format(n)).astype(str)  # .sum()
# df_AUCs_p

# df_AUCs_p.columns

df_AUCs_p.loc[:, [('AUC', 'mean'), ('micro F1-score', 'mean'), ('weighted F1-score', 'mean'),
                  ('tr AUC', 'mean'),
                  ('tr micro F1-score', 'mean'),
                  ('tr weighted F1-score', 'mean')]]

for col in cols_rep:  # ['AUC', 'micro F1-score', 'weighted F1-score']:
    df_AUCs_p[col +
              '_'] = df_AUCs_p[col].apply(lambda x: ' ('.join(x) + ')', axis=1)

[col + '_' for col in cols_rep]

# df_AUCs_p.reindex(axis=0).columns#[col]['mean']

df_AUCs_p[col + '_'].values

# df.columns =
df_AUCs_p2 = df_AUCs_p[[col + '_' for col in cols_rep]]
df_AUCs_p2.columns = cols_rep  # df_AUCs_p2.columns.droplevel(1)
df_AUCs_p2[[x for x in df_AUCs_p2.columns for i in range(2)]]

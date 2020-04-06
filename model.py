# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

## Load libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
import scipy.stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, f1_score, auc
import xgboost as xgb
from sklearn.utils import resample
import skopt
from functools import partial
import shap
import matplotlib.patches as mpatches
from optimalDS import *

# +
import numpy as np
import pandas as pd
import re
import scipy.stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, f1_score, auc
import xgboost as xgb
from sklearn.utils import resample
import skopt
from functools import partial


def pivot_data(data, values, index, columns, reset_index, prefix):
    data = pd.pivot_table(data, values=values, index=index, columns=columns, aggfunc='max', fill_value=None)
    data.columns = data.columns.to_series().str.join('_')
    if reset_index: data = data.reset_index(reset_index)
    data = data.add_prefix(prefix)
    return data


def add_meta_features(data_meta, DieX_name, DieY_name):
    center_x = int((data_meta[DieX_name].max() - data_meta[DieX_name].min()) / 2)
    center_y = int((data_meta[DieY_name].max() - data_meta[DieY_name].min()) / 2)
    data_meta['meta_Radius'] = np.sqrt(
        (data_meta[DieX_name] - center_x) ** 2 + (data_meta[DieY_name] - center_y) ** 2).astype(int)
    return data_meta


def drop_missing_values(data_tests_pivot, Threshold):
    percent_missing = data_tests_pivot.isnull().sum() * 100 / len(data_tests_pivot)
    columns_drop = data_tests_pivot[percent_missing[percent_missing > Threshold].index.to_list()].columns.to_list()
    data_tests_pivot = data_tests_pivot[percent_missing[percent_missing < Threshold].index.to_list()]
    percent_missing_per_wafer = data_tests_pivot.groupby(['Wafer_ID']).apply(lambda x: x.isnull().sum() * 100 / len(x))
    wafers_with_missing_tests = percent_missing_per_wafer.groupby(['Wafer_ID']).apply(lambda x: (x == 100).sum().sum())
    wafers_drop = wafers_with_missing_tests[wafers_with_missing_tests < 0].index.to_list()
    wafers_with_missing_tests = wafers_with_missing_tests[wafers_with_missing_tests > 0].index.to_list()
    data_tests_pivot = data_tests_pivot[
        ~data_tests_pivot.index.get_level_values('Wafer_ID').isin(wafers_with_missing_tests)]
    return data_tests_pivot, columns_drop, wafers_drop


def replace_missing_values(data, impute_grouping, impute_features, missing):
    if missing: missing_data = data.isnull()
    data = data.groupby(impute_grouping)[impute_features].transform(lambda x: x.fillna(x.median()))
    if missing: data = data.merge(missing_data, how='inner', left_index=True, right_index=True,
                                  suffixes=('', '_imputed'))
    return data


def robust_sigma(x, axis=None, rng=(25, 75), scale='normal', nan_policy='omit', interpolation='nearest',
                 keepdims=False):
    rs = scipy.stats.iqr(x, axis=axis, rng=rng, scale=scale, nan_policy=nan_policy, interpolation=interpolation,
                         keepdims=keepdims)
    return (rs)


def stats(data_tests_pivot_imputed, groupby, func, prefix_stats):
    wafer_medians = data_tests_pivot_imputed[
        data_tests_pivot_imputed.columns[~pd.Series(data_tests_pivot_imputed.columns).str.endswith('_imputed')]]
    impute_features = wafer_medians.columns[pd.Series(wafer_medians.columns).str.startswith('tests')]
    data_tests_wafer_medians = wafer_medians.groupby(groupby)[impute_features].agg(func)
    data_tests_wafer_medians.columns = data_tests_wafer_medians.columns.to_series().str.join(prefix_stats)
    return data_tests_wafer_medians


def prep_data_meta(data_meta, values, index, columns, reset_index=None, prefix='Unknown_', radius=None):
    data_model_meta = pivot_data(data_meta, values, index, columns, reset_index, prefix)
    if radius: data_model_meta = add_meta_features(data_model_meta, radius[0], radius[1])
    data_model_meta.to_csv('data_model_meta.csv', index=True)
    return data_model_meta


def prep_data_tests(data_tests, values, index, columns, reset_index=None, prefix='Unknown_',
                    impute_grouping=['Wafer_ID'], cont_func=['median'], missing=True,
                    Threshold=20):
    data_pivot_tests = pivot_data(data_tests, values, index, columns, reset_index, prefix)
    data_tests_pivot_imputed, columns_drop, wafers_drop = drop_missing_values(data_pivot_tests, Threshold)
    data_tests_pivot_imputed = replace_missing_values(data_tests_pivot_imputed, impute_grouping,
                                                      data_tests_pivot_imputed.columns, missing)
    data_tests_wafer_medians = stats(data_tests_pivot_imputed, ['Wafer_ID'], cont_func, '_wafer_')
    data_tests_fablot_medians = stats(data_tests_pivot_imputed, ['FabLot', 'Wafer_ID'], cont_func, '_fablot_')
    data_tests_stats = pd.merge(data_tests_wafer_medians.reset_index(),
                                data_tests_fablot_medians.reset_index(),
                                on=['Wafer_ID'], how='inner').set_index(['FabLot', 'Wafer_ID'])
    data_tests.to_csv('data_model_tests.csv', index=True)
    data_tests_stats.to_csv('data_model_tests_stats.csv', index=True)
    return data_tests_pivot_imputed, columns_drop, wafers_drop, data_tests_stats


def prep_data_output(data_output, good_hard_bins):
    data_output.reset_index()
    for i in range(len(data_output)):
        if data_output.at[i, 'HardBin'] in good_hard_bins:
            data_output.at[i, 'output_Result'] = 'Pass'
        else:
            data_output.at[i, 'output_Result'] = 'Fail'
    data_output.to_csv('data_model_output.csv', index=True)
    return data_output


def merge_all_data(data_meta, data_tests, data_tests_stats, data_output):
    model_input_data = data_meta.reset_index().merge(data_tests.reset_index(),
                                                     how='inner',
                                                     left_on=['FabLot', 'Wafer_ID', 'Unit_ID'],
                                                     right_on=['FabLot', 'Wafer_ID', 'Unit_ID']
                                                     ).merge(data_tests_stats.reset_index(),
                                                             how='inner',
                                                             left_on=['FabLot', 'Wafer_ID'],
                                                             right_on=['FabLot', 'Wafer_ID']
                                                             ).merge(data_output,
                                                                     how='inner',
                                                                     left_on=['FabLot', 'Wafer_ID', 'Unit_ID'],
                                                                     right_on=['FabLot', 'Wafer_ID', 'Unit_ID']
                                                                     ).set_index('Unit_ID'
                                                                                 ).drop(
        ['FabLot', 'Wafer_ID', 'HardBin', 'SoftBin'], axis=1)
    model_input_data = model_input_data.filter(regex='^((?!StartTime).)*$', axis=1)
    return model_input_data


def split_vals(a, n): return a[:n], a[n:]


def change_columns_to_categorical(df, cols):
    for c in cols:
        df[c] = df[c].astype('category')
    return df


def data_balance(model_input_data, percentage_samples, minority, majority):
    # Mixsample balancing
    n_trn = int(model_input_data.shape[0] * percentage_samples)
    x_data_train, x_data_valid = split_vals(model_input_data, n_trn)

    # Separate majority and minority classes
    train_majority = x_data_train[x_data_train['output_Result'] == 'Pass']
    train_minority = x_data_train[x_data_train['output_Result'] == 'Fail']
    resample_count_minority = int(float(train_minority.shape[0]) * minority)
    resample_count_majority = resample_count_minority * majority

    # Downsample majority class
    train_majority_downsampled = resample(train_majority,
                                          replace=False,  # sample with replacement
                                          n_samples=resample_count_majority,  # to match majority class
                                          random_state=42)  # reproducible results

    # Upsample minority class
    train_minority_upsampled = resample(train_minority,
                                        replace=True,  # sample with replacement
                                        n_samples=resample_count_minority,  # to match majority class
                                        random_state=42)  # reproducible results

    # Combine minority class with downsampled majority class
    train_mixedsampled = pd.concat([train_majority_downsampled, train_minority_upsampled])

    x_train = train_mixedsampled.drop('output_Result', axis=1)
    y_train = train_mixedsampled['output_Result']
    x_valid = x_data_valid.drop('output_Result', axis=1)
    y_valid = x_data_valid['output_Result']
    return x_train, y_train, x_valid, y_valid


# function to fit the model and return the performance of the model
def calculate_auc_score(prediction_scores, actual_scores):
    from sklearn import metrics
    import pandas as pd
    import numpy as np
    y_score = pd.DataFrame(prediction_scores, columns=['Fail', 'Pass'])
    y_true = pd.get_dummies(actual_scores)
    model_auc = metrics.roc_auc_score(y_true.Pass, y_score.Pass, average='macro', sample_weight=None, max_fpr=None)
    return model_auc


def calculate_reduction_auc_score(validation_data, target_reduction_percent):
    from sklearn import metrics
    import pandas as pd
    import numpy as np
    reduction_escape_list = []
    reduction_incrament_list = list(np.arange(0, target_reduction_percent + 1e-6, 0.1).round(1))
    num_fails = validation_data[validation_data['Actual'] == 0].shape[0]

    for i in reduction_incrament_list:
        validation_data_i = validation_data[:int(validation_data.shape[0] * i)]
        num_fails_i = validation_data_i[validation_data_i['Actual'] == 0].shape[0]
        reduction_escape_list.append(num_fails_i / num_fails)

    reduction_auc = auc(reduction_incrament_list, reduction_escape_list)

    return reduction_auc


def calculate_escapes_count(validation_data, target_reduction_percent):
    validation_data_i = validation_data[:int(validation_data.shape[0] * target_reduction_percent)]
    num_escapes = validation_data_i[validation_data_i['Actual'] == 0].shape[0]
    return num_escapes


def create_reduction_table(validation_data):
    fail_count = validation_data[validation_data['Actual'] == 0].shape[0]
    pass_count = validation_data[validation_data['Actual'] == 1].shape[0]

    # Create reduciton table
    perc_0 = {'lowest_probability_passed': [1], 'random_select': [0], 'escapes': [0], 'DPPM': [0], 'lift': [1.]}
    reduction_table = pd.DataFrame(perc_0)
    for i in range(1, 11):
        perc = i / 10.
        new_row = {}
        validation_data_i = validation_data[:int(validation_data.shape[0] * perc)]
        new_row['lowest_probability_passed'] = validation_data_i.iloc[-1]['Pass']
        new_row['random_select'] = perc * fail_count
        new_row['escapes'] = validation_data_i[validation_data_i['Actual'] == 0].shape[0]
        new_row['dppm'] = int(new_row['escapes'] * 1e6 / pass_count)

        if new_row['escapes']:  # if different than zero
            new_row['lift'] = new_row['random_select'] / new_row['escapes']
        else:
            new_row['lift'] = np.inf
        new_row = pd.Series(new_row, name=str(i * 10) + '%')
        reduction_table = reduction_table.append(new_row)
    return reduction_table


def create_feature_importance(model, features_list):
    import pandas as pd
    import numpy as np
    model_feature_importance = pd.DataFrame(model.feature_importances_,
                                            index=features_list,
                                            columns=['importance']).sort_values('importance', ascending=False)
    return model_feature_importance


def model_assessment(args, model_type, criteria, x_train, y_train, x_valid, y_valid):
    global models, reduction_table, curr_model_hyper_params, auc_scores, auc_reduction, target_reduction_percent
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    import xgboost as xgb
    params = {curr_model_hyper_params[i]: args[i] for i, j in enumerate(curr_model_hyper_params)}

    if model_type == 'XGBoost':
        model = xgb.XGBClassifier(objective="binary:logistic",
                                  random_state=42,
                                  n_jobs=2)
        model.set_params(**params)
    elif model_type == 'RF':
        model = RandomForestClassifier(criterion='entropy',
                                       bootstrap=True,
                                       oob_score=True,
                                       n_jobs=2,
                                       random_state=42)
        model.set_params(**params)

    fitted_model = model.fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    y_pred_valid = model.predict(x_valid)
    y_pred_train_prob = model.predict_proba(x_train)
    y_pred_valid_prob = model.predict_proba(x_valid)

    # Create validation results table
    y_score = pd.DataFrame(y_pred_valid_prob, columns=['Fail', 'Pass'])
    validation_results = pd.concat([y_valid.to_frame().reset_index()['output_Result'], y_score], axis=1)
    validation_results['Actual'] = np.where(validation_results['output_Result'] == 'Pass', 1, 0)
    validation_results.drop('output_Result', axis=1, inplace=True)
    validation_results.sort_values(['Pass', 'Actual'], ascending=[False, True], inplace=True)

    # Get model feature importance, Reduction Table, AUC score, reduction table AUC score, escapes count at target percentage
    features_list = x_train.columns
    model_feature_importance = create_feature_importance(model, features_list)
    model_reduction_table = create_reduction_table(validation_results)
    model_auc = calculate_auc_score(y_pred_valid_prob, y_valid)
    model_reduction_auc = calculate_reduction_auc_score(validation_results, target_reduction_percent)
    model_escapes_at_target = calculate_escapes_count(validation_results, target_reduction_percent)

    # Creating the various metrics list
    model_metrics_i = {'auc_scores': model_auc, 'auc_reduction': model_reduction_auc,
                       'escapes': model_escapes_at_target}
    for k in model_metrics.keys():
        model_metrics[k].append(model_metrics_i[k])

    # Saving the various metrics and model
    models.append(fitted_model)
    reduction_table.append(model_reduction_table)
    feature_importances.append(model_feature_importance)

    # Select which metric to return
    return_metric = model_metrics_i[criteria]
    return return_metric


def run_model(model_type, target_percent, x_train, y_train, x_valid, y_valid):
    global target_reduction_percent, models, reduction_table, curr_model_hyper_params, auc_scores, auc_reduction, model_metrics, feature_importances
    target_reduction_percent = target_percent

    # Define parameter space for each model type
    space_xgb = [
        skopt.space.Real(0.5, 1.0, name="colsample_bylevel"),
        skopt.space.Real(0.5, 1.0, name="colsample_bytree"),
        skopt.space.Real(0, 20, name="gamma"),
        skopt.space.Real(0.00001, 1, name="learning_rate"),
        skopt.space.Real(0.1, 20, name="max_delta_step"),
        skopt.space.Integer(3, 9, name="max_depth"),
        skopt.space.Real(10, 500, name="min_child_weight"),
        skopt.space.Integer(10, 2000, name="n_estimators"),
        skopt.space.Real(0.00001, 1.0, name="reg_alpha"),
        skopt.space.Real(0.0001, 10.0, name="reg_lambda"),
        skopt.space.Real(0.5, 1.0, name="subsample"),
        skopt.space.Real(0.5, 1.0, name="scale_pos_weight")
    ]

    space_rf = [
        #     skopt.space.Real(1, 15, name="max_depth"),
        #     skopt.space.Real(0.0, 0.5, name="min_weight_fraction_leaf"),
        #     skopt.space.Integer(3, 9, name=" max_leaf_nodes"),
        #     skopt.space.Real(0.0, 10, name="min_impurity_decrease"),
        #     skopt.space.Real(0.5, 1.0, name="class_weight"),
        skopt.space.Integer(1, 300, name="n_estimators"),
        skopt.space.Integer(2, 20, name="min_samples_split"),
        skopt.space.Integer(1, 10, name="min_samples_leaf"),
        skopt.space.Real(0.1, 0.7, name="max_features")
    ]
    if model_type == 'XGBoost':
        space = space_xgb
        curr_model_hyper_params = ['colsample_bylevel', 'colsample_bytree', 'gamma', 'learning_rate', 'max_delta_step',
                                   'max_depth', 'min_child_weight', 'n_estimators', 'reg_alpha', 'reg_lambda',
                                   'subsample',
                                   "scale_pos_weight"]
    elif model_type == 'RF':
        space = space_rf
        curr_model_hyper_params = ['n_estimators', 'min_samples_split', 'min_samples_leaf', 'max_features']

    # Create container lists
    models = []
    reduction_table = []
    feature_importances = []
    model_metrics = {'auc_scores': [], 'auc_reduction': [], 'escapes': []}

    # Determine the working funciton
    objective_function = partial(model_assessment,
                                 model_type=model_type,
                                 criteria='auc_reduction',
                                 x_train=x_train,
                                 y_train=y_train,
                                 x_valid=x_valid,
                                 y_valid=y_valid)

    # Running the optimization algorithm
    results = skopt.gp_minimize(objective_function,
                                space,
                                base_estimator=None,
                                n_calls=100,
                                n_random_starts=5,
                                random_state=42,
                                verbose=True)
    all_iteration_results = pd.DataFrame(results.x_iters, columns=curr_model_hyper_params)
    skopt_results = pd.DataFrame({'AUC': model_metrics['auc_scores'],
                                  'AUC_Reduction': model_metrics['auc_reduction'],
                                  'Escapes': model_metrics['escapes']})
    skopt_results = pd.concat([skopt_results, all_iteration_results], axis=1)
    skopt_results.index.name = 'Iteration'
    model = models[
        skopt_results.loc[skopt_results['AUC_Reduction'] == min(skopt_results['AUC_Reduction'])].reset_index().iloc[
            0, 0]]
    predict_proba = pd.DataFrame(model.predict_proba(x_train))
    fails = x_train.copy()
    fails['predict_proba'] = 0
    fails['p/f'] = y_train
    fails = fails.reset_index()
    for i in range(len(fails)):
        fails.loc[i, 'predict_proba'] = predict_proba.loc[i][0]
    fails = fails.sort_values(by=['predict_proba'])
    fails = fails[:int(fails.shape[0] * (target_reduction_percent))]
    fails = fails.set_index('Unit_ID')
    fails = fails.drop(columns=['predict_proba'])
    fails = fails.loc[fails.loc[:, 'p/f'] == 'Fail']
    fails = fails.drop('p/f', axis=1)
    return skopt_results, models, model, fails


def shap_js(model, value):
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(value)
    return shap.force_plot(explainer.expected_value[0], shap_values[0], value)


def shap_js_bar_plot(model, values):
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(values)
    return shap.summary_plot(shap_values, values, plot_type="bar")


def drop_bad_columns(threshold, fails, model, x_train, x_valid):
    sampling = x_train
    if str(type(model)) == "<class 'sklearn.ensemble._forest.RandomForestClassifier'>":
        sampling = x_train[:500]
    explainer = shap.TreeExplainer(model)
    shap_values_fails = explainer.shap_values(fails)
    shap_values_data = explainer.shap_values(sampling)

    if str(type(model)) == "<class 'sklearn.ensemble._forest.RandomForestClassifier'>":
        shap_values_fails = shap_values_fails[0]
        shap_values_data = shap_values_data[0]

    shap_values_fails = pd.DataFrame(shap_values_fails, columns=fails.columns.values).sum() / len(shap_values_fails)
    shap_values_data = pd.DataFrame(shap_values_data, columns=fails.columns.values).sum() / len(shap_values_data)
    shap_values_to_drop = (shap_values_fails - shap_values_data)
    columns_to_drop = shap_values_to_drop.loc[shap_values_to_drop > threshold].reset_index()
    x_train = x_train.drop(columns_to_drop['index'].tolist(), axis=1)
    x_valid = x_valid.drop(columns_to_drop['index'].tolist(), axis=1)
    return x_train, x_valid


# -

## Load data
data_meta = pd.read_csv('data_meta.csv', index_col=['FabLot', 'Wafer_ID', 'Unit_ID', 'DieX', 'DieY'])
data_tests = pd.read_csv('data_tests.csv', index_col=['FabLot', 'Wafer_ID', 'Unit_ID'])
data_output = pd.read_csv('data_output.csv', index_col=['Unit_ID'])

# prep data meta
data_model_meta = prep_data_meta(data_meta=data_meta,
                                 values=['StartTime', 'HardBin', 'SoftBin', 'DieTestOrder', 'TouchDownSeq'],
                                 index=['FabLot', 'Wafer_ID', 'Unit_ID', 'DieX', 'DieY', 'Segment', 'Ring'],
                                 columns=['Operation'],
                                 reset_index=['DieX', 'DieY', 'Segment', 'Ring'],
                                 prefix='meta_',
                                 radius=['meta_DieX', 'meta_DieY'])

# prep data tests and create data stats
data_model_tests, columns_drop, wafers_drop, data_model_tests_stats = prep_data_tests(data_tests=data_tests,
                                                                                      values=['FloatValue'],
                                                                                      index=['FabLot', 'Wafer_ID',
                                                                                             'Unit_ID'],
                                                                                      columns=['ParametricTestName',
                                                                                               'Operation'],
                                                                                      prefix='tests_',
                                                                                      impute_grouping=['FabLot',
                                                                                                       'Wafer_ID'],
                                                                                      cont_func=['mean', 'median',
                                                                                                 robust_sigma],
                                                                                      missing=True)

# prep data output
data_model_output = prep_data_output(data_output.reset_index(), [0])

# merge all data
model_input_data = merge_all_data(data_model_meta, data_model_tests, data_model_tests_stats, data_model_output)

# data balance and split
x_train, y_train, x_valid, y_valid = data_balance(model_input_data, 0.77, 10, 10)

# run model and get results
results1, models, model, fails = run_model('XGBoost', 0.3, x_train, y_train, x_valid, y_valid);
results1

x_train, x_valid = drop_bad_columns(0.1, fails, model, x_train, x_valid)

results2, models, model, fails = run_model('XGBoost', 0.3, x_train, y_train, x_valid, y_valid);
results2

shap_js_bar_plot(model, fails)

shap_js_bar_plot(model, x_train)

# +
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 9))
blue_patch = mpatches.Patch(color='blue', label='Original RF models')
orange_patch = mpatches.Patch(color='orange', label='New RF models')
green_patch = mpatches.Patch(color='green', label='Original XGboost models')
red_patch = mpatches.Patch(color='red', label='New XGboost models')
sns.lineplot(x=range(len(results3)), y='AUC_Reduction', marker='o',
             data=results3)
sns.lineplot(x=range(len(results4)), y='AUC_Reduction', marker='o',
             data=results4)
sns.lineplot(x=range(len(results)), y='AUC_Reduction', marker='o',
             data=results)
sns.lineplot(x=range(len(results2)), y='AUC_Reduction', marker='o',
             data=results2)
plt.legend(handles=[blue_patch, orange_patch, green_patch, red_patch])

plt.title("AUC Reduction in Models", fontsize=20)
plt.xlabel("Model Number", fontsize=15)
plt.ylabel("AUC Reduction", fontsize=15)
# -

x_train, y_train, x_valid, y_valid = data_balance(model_input_data, 0.77, 10, 10)
predict_proba = pd.DataFrame(best_random.predict_proba(x_valid))
fails = x_valid.copy()
fails['predict_proba'] = 0
fails['p/f'] = y_valid
fails = fails.reset_index()
for i in range(len(fails)):
    fails.loc[i, 'predict_proba'] = predict_proba.loc[i][0]
fails = fails.sort_values(by=['predict_proba'])
fails = fails[:int(fails.shape[0] * (target_reduction_percent))]
fails = fails.set_index('Unit_ID')
fails = fails.drop(columns=['predict_proba'])
fails = fails.loc[fails.loc[:, 'p/f'] == 'Fail']
fails = fails.drop('p/f', axis=1)

results1.loc[54]

results1.loc[results1['AUC_Reduction'] == min(results1['AUC_Reduction'])]['AUC_Reduction']

results2.loc[results2['AUC_Reduction'] == min(results2['AUC_Reduction'])]['AUC_Reduction']

results3.loc[results3['AUC_Reduction'] == min(results3['AUC_Reduction'])]['AUC_Reduction']

results4.loc[7, 'AUC_Reduction'] = 0.02195

results3.loc[results3['AUC_Reduction'] == min(results3['AUC_Reduction'])]['AUC_Reduction']

shap.initjs()
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(fails.iloc[4])
shap.force_plot(explainer.expected_value[0], shap_values[0], fails.iloc[4])

predict_proba = pd.DataFrame(model.predict_proba((x_valid)))
fails = x_valid.copy()
fails['predict_proba'] = 0
fails['p/f'] = y_valid
fails = fails.reset_index()
for i in range(len(fails)):
    fails.loc[i, 'predict_proba'] = predict_proba.loc[i][0]
fails = fails.sort_values(by=['predict_proba'])
fails = fails[:int(fails.shape[0] * (target_reduction_percent))]
fails = fails.set_index('Unit_ID')
fails = fails.drop(columns=['predict_proba'])
fails = fails.loc[fails.loc[:, 'p/f'] == 'Fail']
fails = fails.drop('p/f', axis=1)

x_train, y_train, x_valid, y_valid = data_balance(model_input_data, 0.77, 10, 10)
y_train = y_train.replace(['Pass', 'Fail'], [1, 0])
y_valid = y_valid.replace(['Pass', 'Fail'], [1, 0])
model = RandomForestClassifier(
    oob_score=True,
    n_jobs=-1)
model.fit(x_train, y_train)
predictions = model.predict(x_valid)

x_valid['predictions'] = predictions

x_valid['predictions'].value_counts()

x_train, y_train, x_valid, y_valid = data_balance(model_input_data, 0.77, 10, 10)
predict_proba = pd.DataFrame(model.predict_proba(x_valid))
fails = x_valid.copy()
fails['p/f'] = y_valid
fails = fails.reset_index()
for i in range(len(fails)):
    fails.loc[i, 'predict_proba'] = predict_proba.loc[i][0]
fails = fails.sort_values(by=['predict_proba'])
fails = fails[:int(fails.shape[0] * (target_reduction_percent))]
fails = fails.set_index('Unit_ID')
fails = fails.drop(columns=['predict_proba'])
fails = fails.loc[fails.loc[:, 'p/f'] == 'Fail']
fails = fails.drop('p/f', axis=1)

x_train, y_train, x_valid, y_valid = data_balance(model_input_data, 0.77, 20, 5)
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestClassifier(random_state=42)
from pprint import pprint

# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())

from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=5, cv=3, verbose=2,
                               random_state=42, n_jobs=-1)
# Fit the random search model
rf_random.fit(x_train, y_train)

x_train, y_train, x_valid, y_valid = data_balance(model_input_data, 0.77, 20, 5)
best_random.fit(x_train, y_train)
predictions = best_random.predict(x_train)

rf_random.best_params_

best_random

predictions = pd.Series(predictions).value_counts()

predictions


# +
def calculate_auc_score(prediction_scores, actual_scores):
    from sklearn import metrics
    import pandas as pd
    import numpy as np
    y_score = pd.DataFrame(prediction_scores, columns=['Fail', 'Pass'])
    y_true = pd.get_dummies(actual_scores)
    model_auc = metrics.roc_auc_score(y_true.Pass, y_score.Pass, average='macro', sample_weight=None, max_fpr=None)
    return model_auc


def calculate_reduction_auc_score(validation_data, target_reduction_percent):
    from sklearn import metrics
    import pandas as pd
    import numpy as np
    reduction_escape_list = []
    reduction_incrament_list = list(np.arange(0, target_reduction_percent + 1e-6, 0.1).round(1))
    num_fails = validation_data[validation_data['Actual'] == 0].shape[0]

    for i in reduction_incrament_list:
        validation_data_i = validation_data[:int(validation_data.shape[0] * i)]
        num_fails_i = validation_data_i[validation_data_i['Actual'] == 0].shape[0]
        reduction_escape_list.append(num_fails_i / num_fails)

    reduction_auc = auc(reduction_incrament_list, reduction_escape_list)

    return reduction_auc


def calculate_escapes_count(validation_data, target_reduction_percent):
    validation_data_i = validation_data[:int(validation_data.shape[0] * target_reduction_percent)]
    num_escapes = validation_data_i[validation_data_i['Actual'] == 0].shape[0]
    return num_escapes


def create_reduction_table(validation_data):
    fail_count = validation_data[validation_data['Actual'] == 0].shape[0]
    pass_count = validation_data[validation_data['Actual'] == 1].shape[0]

    # Create reduciton table
    perc_0 = {'lowest_probability_passed': [1], 'random_select': [0], 'escapes': [0], 'DPPM': [0], 'lift': [1.]}
    reduction_table = pd.DataFrame(perc_0)
    for i in range(1, 11):
        perc = i / 10.
        new_row = {}
        validation_data_i = validation_data[:int(validation_data.shape[0] * perc)]
        new_row['lowest_probability_passed'] = validation_data_i.iloc[-1]['Pass']
        new_row['random_select'] = perc * fail_count
        new_row['escapes'] = validation_data_i[validation_data_i['Actual'] == 0].shape[0]
        new_row['dppm'] = int(new_row['escapes'] * 1e6 / pass_count)

        if new_row['escapes']:  # if different than zero
            new_row['lift'] = new_row['random_select'] / new_row['escapes']
        else:
            new_row['lift'] = np.inf
        new_row = pd.Series(new_row, name=str(i * 10) + '%')
        reduction_table = reduction_table.append(new_row)
    return reduction_table


model = RandomForestClassifier(criterion='entropy',
                               max_features=15,
                               min_samples_split=9,
                               n_estimators=288)
fitted_model = model.fit(x_train, y_train)

y_pred_train = model.predict(x_train)
y_pred_valid = model.predict(x_valid)
y_pred_train_prob = model.predict_proba(x_train)
y_pred_valid_prob = model.predict_proba(x_valid)

# Create validation results table
y_score = pd.DataFrame(y_pred_valid_prob, columns=['Fail', 'Pass'])
validation_results = pd.concat([y_valid.to_frame().reset_index()['output_Result'], y_score], axis=1)
validation_results['Actual'] = np.where(validation_results['output_Result'] == 'Pass', 1, 0)
validation_results.drop('output_Result', axis=1, inplace=True)
validation_results.sort_values(['Pass', 'Actual'], ascending=[False, True], inplace=True)

# Get model feature importance, Reduction Table, AUC score, reduction table AUC score, escapes count at target percentage
features_list = x_train.columns
model_feature_importance = create_feature_importance(model, features_list)
model_reduction_table = create_reduction_table(validation_results)
model_auc = calculate_auc_score(y_pred_valid_prob, y_valid)
model_reduction_auc = calculate_reduction_auc_score(validation_results, target_reduction_percent)
model_escapes_at_target = calculate_escapes_count(validation_results, target_reduction_percent)

# Creating the various metrics list
model_metrics_i = {'auc_scores': model_auc, 'auc_reduction': model_reduction_auc, 'escapes': model_escapes_at_target}
for k in model_metrics.keys():
    model_metrics[k].append(model_metrics_i[k])

# Saving the various metrics and model
models.append(fitted_model)
reduction_table.append(model_reduction_table)
feature_importances.append(model_feature_importance)
# -

model_reduction_auc

# +
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint

param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, x_train.shape[1]),
              "min_samples_split": sp_randint(2, 11),
              "bootstrap": [True, False],
              "n_estimators": sp_randint(100, 500)}

random_search = RandomizedSearchCV(rf, param_distributions=param_dist,
                                   n_iter=2, cv=3, iid=False, random_state=42, scoring='roc_auc')
random_search.fit(x_train, y_train)
# -

random_search.best_params_

# +
from sklearn.metrics.scorer import make_scorer
from sklearn import metrics

target_reduction_percent = 0.3

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint

my_scorer = make_scorer(calculate_reduction_auc_score, greater_is_better=False, needs_proba=True)

param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, x_train.shape[1]),
              "min_samples_split": sp_randint(2, 11),
              "bootstrap": [True, False],
              "n_estimators": sp_randint(100, 500)}

random_search = RandomizedSearchCV(rf, param_distributions=param_dist,
                                   n_iter=2, cv=3, iid=False, random_state=42, scoring=my_scorer)
random_search.fit(x_train, y_train)


# -

def calculate_reduction_auc_score(estimator, X, y):
    y_pred_train = estimator.predict(x_train)
    y_pred_valid = estimator.predict(x_valid)
    y_pred_train_prob = estimator.predict_proba(x_train)
    y_pred_valid_prob = estimator.predict_proba(x_valid)

    # Create validation results table
    y_score = pd.DataFrame(y_pred_valid_prob, columns=['Fail', 'Pass'])
    validation_results = pd.concat([y_valid.to_frame().reset_index()['output_Result'], y_score], axis=1)
    validation_results['Actual'] = np.where(validation_results['output_Result'] == 'Pass', 1, 0)
    validation_results.drop('output_Result', axis=1, inplace=True)
    validation_results.sort_values(['Pass', 'Actual'], ascending=[False, True], inplace=True)
    reduction_escape_list = []
    reduction_incrament_list = list(np.arange(0, target_reduction_percent + 1e-6, 0.1).round(1))
    num_fails = validation_results[validation_results['Actual'] == 0].shape[0]

    for i in reduction_incrament_list:
        validation_data_i = validation_results[:int(validation_results.shape[0] * i)]
        num_fails_i = validation_data_i[validation_data_i['Actual'] == 0].shape[0]
        reduction_escape_list.append(num_fails_i / num_fails)

    reduction_auc = auc(reduction_incrament_list, reduction_escape_list)
    print(reduction_auc)
    return reduction_auc


y_valid = y_valid[:int(y_valid.shape[0] * (target_reduction_percent))]

x_train = model.predict(x_train)
x_train = x_train[:int(x_train.shape[0] * (target_reduction_percent))]

# +
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
import pandas as pd
import numpy as np


def my_custom_loss_func_est(estimator, X, y):
    y_pred_train = estimator.predict(x_train)
    y_pred_valid = estimator.predict(x_valid)
    y_pred_train_prob = estimator.predict_proba(x_train)
    y_pred_valid_prob = estimator.predict_proba(x_valid)

    # Create validation results table
    y_score = pd.DataFrame(y_pred_valid_prob, columns=['Fail', 'Pass'])
    validation_results = pd.concat([y_valid.to_frame().reset_index()['output_Result'], y_score], axis=1)
    validation_results['Actual'] = np.where(validation_results['output_Result'] == 'Pass', 1, 0)
    validation_results.drop('output_Result', axis=1, inplace=True)
    validation_results.sort_values(['Pass', 'Actual'], ascending=[False, True], inplace=True)
    reduction_escape_list = []
    reduction_incrament_list = list(np.arange(0, target_reduction_percent + 1e-6, 0.1).round(1))
    num_fails = validation_results[validation_results['Actual'] == 0].shape[0]

    for i in reduction_incrament_list:
        validation_data_i = validation_results[:int(validation_results.shape[0] * i)]
        num_fails_i = validation_data_i[validation_data_i['Actual'] == 0].shape[0]
        reduction_escape_list.append(num_fails_i / num_fails)

    reduction_auc = auc(reduction_incrament_list, reduction_escape_list)
    print(reduction_auc)
    return reduction_auc


custom_scorer = make_scorer(my_custom_loss_func,
                            greater_is_better=True,
                            needs_proba=True)

param_dist = {"max_depth": [3, None],
              "max_features": [1, 50],
              "min_samples_split": [2, 15],
              "bootstrap": [True, False],
              "n_estimators": [100, 500]}
# -

grid = GridSearchCV(RandomForestClassifier(), param_grid=param_dist,
                    scoring=my_custom_loss_func_est, return_train_score=True)
grid_result = grid.fit(x_train, y_train.replace(['Pass', 'Fail'], [1, 0]))
results = pd.DataFrame(grid.cv_results_)[['param_bootstrap',
                                          'param_max_depth',
                                          'param_max_features',
                                          'param_min_samples_split',
                                          'param_n_estimators',
                                          'mean_test_score']]

grid_result.best_params_

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=20,
                               cv=3, verbose=2, random_state=42, n_jobs=-1, scoring=my_custom_loss_func_est)
grid_result = rf_random.fit(x_train, y_train.replace(['Pass', 'Fail'], [1, 0]))

rf_random.best_params_

from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)

model = RandomForestClassifier(n_estimators=200,
                               min_samples_split=5,
                               min_samples_leaf=2,
                               max_features='sqrt',
                               max_depth=10,
                               bootstrap=True)
model.fit(x_train, y_train)
predictions = model.predict(x_valid)
print('AUC_Reduction = ' + str(calculate_reduction_auc_score(model, y_valid, predictions)))

a = type(model);
str(a) == "<class 'sklearn.ensemble._forest.RandomForestClassifier'>"

x = [0.0475, 0.021, 0.0183, 0.009917]
y = pd.DataFrame(x);
y['model number'] = range(1, 5);
y
y['AUC Reduction'] = y[0];
y

plt.figure(figsize=(16, 9))
Text = mpatches.Patch(color='white', label='The most optimal models:')
blue_patch = mpatches.Patch(color='blue', label='Original RF models')
orange_patch = mpatches.Patch(color='orange', label='New RF models')
green_patch = mpatches.Patch(color='green', label='Original XGboost models')
red_patch = mpatches.Patch(color='red', label='New XGboost models')
plt.legend(handles=[Text, blue_patch, orange_patch, green_patch, red_patch])
graph = sns.barplot(x='model number', y='AUC Reduction', data=y)

# +
x_train, y_train, x_valid, y_valid = data_balance(model_input_data, 0.77, 10, 10)
sampling = x_train
if str(type(model)) == "<class 'sklearn.ensemble._forest.RandomForestClassifier'>":
    sampling = x_train[:500]
explainer = shap.TreeExplainer(model)
shap_values_fails = explainer.shap_values(fails)
shap_values_data = explainer.shap_values(sampling)

if str(type(model)) == "<class 'sklearn.ensemble._forest.RandomForestClassifier'>":
    shap_values_fails = shap_values_fails[0]
    shap_values_data = shap_values_data[0]

shap_values_fails = pd.DataFrame(shap_values_fails, columns=fails.columns.values).sum() / len(shap_values_fails)
shap_values_data = pd.DataFrame(shap_values_data, columns=fails.columns.values).sum() / len(shap_values_data)
shap_values_to_drop = (shap_values_fails - shap_values_data)
columns_to_drop = shap_values_to_drop.loc[shap_values_to_drop > 0.001].reset_index()
x_train = x_train.drop(columns_to_drop['index'].tolist(), axis=1)
x_valid = x_valid.drop(columns_to_drop['index'].tolist(), axis=1)
# -


import pickle, os, glob, ast
import os
from utils import *
from models import * 
import yaml
from codecarbon import EmissionsTracker

seed = 42
set_seed(seed)

with open('../configs/config_malanville.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)


print('Config file: {}'.format(config))

"""Extract data params"""
data_params = config['data_params']

B_ARG = data_params['B_ARG']

"""Create a directory to save results"""
save_path = data_params['save_path']

"""Create subdirectories to save results"""
dirs = [data_params['city']+"/Images/",
        data_params['city']+"/Models/",
        data_params['city']+"/Excel/"]

for path in dirs:
    if not os.path.exists(save_path+path):
        os.makedirs(save_path+path)

"""Lagged data path"""
lagged_data_path = data_params['data_path'] + 'lagged/' + data_params['city'] + '_lagged.csv'

target_column = data_params['target_column']

dataset = pd.read_csv(lagged_data_path)

dataset.set_index('DATE', inplace=True)

print(dataset)

"""Convert split_date to datetime"""
split_date = data_params['split_date']

target = dataset[target_column]
dataset.drop(target_column, axis=1,inplace=True)

"""Split the DataFrame"""
X_trains, X_tests  = dataset.loc[:split_date], dataset.loc[split_date:]
y_train, y_test = target.loc[:split_date], target.loc[split_date:]

X_train, X_test = np.array(X_trains), np.array(X_tests)

times = list(dataset.index)

print_banner('Data shape')
print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Test set shape: {X_test.shape}, {y_test.shape}")

tracker = EmissionsTracker(output_dir=os.path.join(os.getcwd(), save_path, data_params['city']),
    output_file='CO2_detailed_output.csv',
    project_name='Linear Regression'
)
tracker.start()

print_banner("Training Linear Regression")

linear_reg_model = conf_model( LinearRegression(), X_train, y_train, mtype='ML', B_ARG=B_ARG)

y_pred1, y_pred_lower, y_pred_upper = linear_reg_model.predict(X_test, alpha=.1, y_true=y_test, s=None)
y_train_pred1, y_train_pred_lower, y_train_pred_upper = linear_reg_model.predict(X_train, alpha=.1, y_true=y_train, s=None)

plot_result(y_train, y_test, y_train_pred1, y_pred1,dates = list(dataset.index), title='Linear Regression')
plt.savefig(save_path + dirs[0] + 'lr_model_with_lags.png')

plot_results_test(y_test, y_pred1, title='Linear Regression')
plt.savefig(save_path + dirs[0] + 'lr_model_with_lags_test_part.png')

plot_predicted_interval(y_test, y_pred1, y_pred_lower, y_pred_upper, "Linear Regression")
plt.savefig(save_path + dirs[0] + 'lr_model_with_lags_test_part_cf.png')

linear_lr_results = evaluate_preds(X_test, y_test, y_pred1, y_pred_lower, y_pred_upper)
emissions = tracker.stop()
linear_lr_results['CO2_EMISSIONS'] = emissions

print("Linear regression results: ", linear_lr_results)



tracker = EmissionsTracker(output_dir=os.path.join(os.getcwd(), save_path, data_params['city']),
    output_file='CO2_detailed_output.csv',
    project_name='Ridge'
)
tracker.start()

print_banner("Training Ridge")

ridge_model = conf_model(Ridge(), X_train, y_train, mtype = "ML", B_ARG=B_ARG)

y_pred2, y_pred_lower, y_pred_upper = ridge_model.predict(X_test, alpha=.1, y_true=y_test, s=None)
y_train_pred2, y_train_pred_lower, y_train_pred_upper = ridge_model.predict(X_train, alpha=.1, y_true=y_train, s=None)

plot_result(y_train, y_test, y_train_pred2, y_pred2,dates = list(dataset.index), title='Ridge')
plt.savefig(save_path + dirs[0] + 'ridge_model_with_lags.png')

plot_results_test(y_test, y_pred2, title='Ridge')
plt.savefig(save_path + dirs[0] + 'ridge_model_with_lags_test_part.png')

plot_predicted_interval(y_test, y_pred2, y_pred_lower, y_pred_upper, "Ridge")
plt.savefig(save_path + dirs[0] + 'ridge_model_with_lags_test_part_cf.png')

ridge_results = evaluate_preds(X_test, y_test, y_pred2, y_pred_lower, y_pred_upper)
emissions = tracker.stop()
ridge_results['CO2_EMISSIONS'] = emissions

print("Ridge results: ", ridge_results)



tracker = EmissionsTracker(output_dir=os.path.join(os.getcwd(), save_path, data_params['city']),
    output_file='CO2_detailed_output.csv',
    project_name='RandomForest'
)
tracker.start()

print_banner("Training RandomForest")

rforest_model = conf_model(RandomForestRegressor(max_depth=3, random_state=0), X_train, y_train, mtype = "ML", B_ARG=B_ARG)

y_pred3, y_pred_lower, y_pred_upper = rforest_model.predict(X_test, alpha=.1, y_true=y_test, s=None)
y_train_pred3, y_train_pred_lower, y_train_pred_upper = rforest_model.predict(X_train, alpha=.1, y_true=y_train, s=None)

plot_result(y_train, y_test, y_train_pred3, y_pred3,dates = list(dataset.index), title='Random Forest')
plt.savefig(save_path + dirs[0] + 'random_forest_model_with_lags.png')

plot_results_test(y_test, y_pred3, title='Random Forest')
plt.savefig(save_path + dirs[0] + 'random_forest_model_with_lags_test_part.png')

plot_predicted_interval(y_test, y_pred3, y_pred_lower, y_pred_upper, "Random Forest")
plt.savefig(save_path + dirs[0] + 'random_forest_model_with_lags_test_part_cf.png')

rforest_results = evaluate_preds(X_test, y_test, y_pred3, y_pred_lower, y_pred_upper)
emissions = tracker.stop()
rforest_results['CO2_EMISSIONS'] = emissions

print("Random forest regressor results: ", rforest_results)



tracker = EmissionsTracker(output_dir=os.path.join(os.getcwd(), save_path, data_params['city']),
    output_file='CO2_detailed_output.csv',
    project_name='Xgboost'
)
tracker.start()

print_banner("Training XGBoost")

xgboost_model = conf_model(XGBRegressor(), X_train, y_train, mtype = "ML", B_ARG=B_ARG)

y_pred4, y_pred_lower, y_pred_upper = xgboost_model.predict(X_test, alpha=.1, y_true=y_test, s=None)

y_train_pred4, y_train_pred_lower, y_train_pred_upper = xgboost_model.predict(X_train, alpha=.1, y_true=y_train, s=None)

plot_result(y_train, y_test, y_train_pred4, y_pred4,dates = list(dataset.index), title='Xgboost')
plt.savefig(save_path + dirs[0] + 'xgb_model_with_lags.png')

plot_results_test(y_test, y_pred4, title='Xgboost')
plt.savefig(save_path + dirs[0] + 'xgb_model_with_lags_test_part.png')

plot_predicted_interval(y_test, y_pred4, y_pred_lower, y_pred_upper, "Xgboost")
plt.savefig(save_path + dirs[0] + 'xgb_model_with_lags_test_part_cf.png')

xgboost_results = evaluate_preds(X_test, y_test, y_pred4, y_pred_lower, y_pred_upper)
emissions = tracker.stop()
xgboost_results['CO2_EMISSIONS'] = emissions

print("Xgboost results: ", xgboost_results)



tracker = EmissionsTracker(output_dir=os.path.join(os.getcwd(), save_path, data_params['city']),
    output_file='CO2_detailed_output.csv',
    project_name='Lightgbm'
)
tracker.start()

print_banner("Training LightGBM")

lightgbm_model = conf_model(LGBMRegressor(verbose=-1), X_train, y_train, mtype = "ML", B_ARG=B_ARG)

y_pred5, y_pred_lower, y_pred_upper = lightgbm_model.predict(X_test, alpha=.1, y_true=y_test, s=None)
y_train_pred5, y_train_pred_lower, y_train_pred_upper = lightgbm_model.predict(X_train, alpha=.1, y_true=y_train, s=None)

plot_result(y_train, y_test, y_train_pred5, y_pred5,dates = list(dataset.index), title='Lightgbm')
plt.savefig(save_path + dirs[0] + 'lgbm_model_with_lags.png')

plot_results_test(y_test, y_pred5, title='Lightgbm')
plt.savefig(save_path + dirs[0] + 'lgbm_model_with_lags_test_part.png')

plot_predicted_interval(y_test, y_pred5, y_pred_lower, y_pred_upper, "Lightgbm")
plt.savefig(save_path + dirs[0] + 'lgbm_model_with_lags_test_part_cf.png')

lightgbm_results = evaluate_preds(X_test, y_test, y_pred5, y_pred_lower, y_pred_upper)
emissions = tracker.stop()
lightgbm_results['CO2_EMISSIONS'] = emissions

print("Lightgbm results: ", lightgbm_results)



tracker = EmissionsTracker(
    output_dir=os.path.join(os.getcwd(), save_path, data_params['city']),
    output_file='CO2_detailed_output.csv',
    project_name='SVR'
)
tracker.start()

print_banner("Training SVR")

svr_model = conf_model(SVR(), X_train, y_train, mtype = "ML", B_ARG=B_ARG)
y_pred6, y_pred_lower, y_pred_upper = svr_model.predict(X_tests, alpha=.1, y_true=y_test, s=None)

y_train_pred6, y_train_pred_lower, y_train_pred_upper = svr_model.predict(X_train, alpha=.1, y_true=y_train, s=None)

plot_result(y_train, y_test, y_train_pred6, y_pred6,dates = list(dataset.index), title='SVR')
plt.savefig(save_path + dirs[0] + 'svr_model_with_lags.png')

plot_results_test(y_test, y_pred6, title='SVR')
plt.savefig(save_path + dirs[0] + 'svr_model_with_lags_test_part.png')

plot_predicted_interval(y_test, y_pred6, y_pred_lower, y_pred_upper, "SVR")
plt.savefig(save_path + dirs[0] + 'SVR_model_with_lags_test_part_cf.png')

svr_results = evaluate_preds(X_test, y_test, y_pred6, y_pred_lower, y_pred_upper)
emissions = tracker.stop()
svr_results['CO2_EMISSIONS'] = emissions

print("SVR results: ", svr_results)



X_train = X_trains.values
X_test = X_tests.values

X_train = X_train.reshape((X_train.shape[0],1,X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0],1,X_test.shape[1]))


"""Calculate the index at which to split for validation data"""
split_index = int(0.8 * len(X_train))


X_train, X_val = X_train.copy()[:split_index], X_train.copy()[split_index:]
y_train, y_val = y_train.copy()[:split_index], y_train.copy()[split_index:]

print_banner('Data shape')
print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Test set shape: {X_test.shape}, {y_test.shape}")


tracker = EmissionsTracker(output_dir=os.path.join(os.getcwd(), save_path, data_params['city']),
    output_file='CO2_detailed_output.csv',
    project_name='Conv1D-model'
)
tracker.start()

print_banner("Training Conv1D model")

conv1d_model_params = config['conv1d_model']
file_name = "conv1D_train_output.csv"
train_output_path = os.path.join(save_path, data_params['city'], file_name)

CNN = CNN_1D(n_filters = conv1d_model_params['cnn_units'],
                dense_layers = conv1d_model_params['dense_layers'],
                kernel_s = 1,
                dense_units = conv1d_model_params['dense_units'],
                input_shape = X_train.shape[1:],
                activations = conv1d_model_params['activ'])


CNNP = conf_model(CNN, X_train, y_train, X_val, y_val, train_output_path, mtype="DL", 
    epochs=conv1d_model_params['epochs'], batch_size=conv1d_model_params['batch_size'], B_ARG=B_ARG
)

Y_pred, y_pred_lower, y_pred_upper = CNNP.predict(X_test, alpha=.1, y_true=y_test, s=None)
conv1D_results = evaluate_preds(X_test, y_test, Y_pred, y_pred_lower, y_pred_upper)
plot_predicted_interval(y_test, Y_pred, y_pred_lower, y_pred_upper, "CNN")
plt.savefig(save_path + dirs[0] + 'cnn_model_with_lags_test_part_cf.png')

Y_train_pred, y_train_pred_lower, y_train_pred_upper = CNNP.predict(X_train, alpha=.1, y_true=y_train, s=None)

plot_result(y_train, y_test, Y_train_pred, Y_pred,dates = list(dataset.index), title='CNN')

emissions = tracker.stop()
conv1D_results['CO2_EMISSIONS'] = emissions

print("CONV1D results: ", conv1D_results)



tracker = EmissionsTracker(
    output_dir=os.path.join(os.getcwd(), save_path, data_params['city']),
    output_file='CO2_detailed_output.csv',
    project_name='LSTM-model'
)
tracker.start()

print_banner("Training LSTM")

lstm_model_params = config['lstm_model']
file_name = "LSTM_train_output.csv"
train_output_path = os.path.join(save_path, data_params['city'], file_name)

lstm_model_instance = flexible_LSTM(
    lstm_layers = lstm_model_params['lstm_layers'],
    hidden_units = lstm_model_params['lstm_units'], 
    dense_layers = lstm_model_params['dense_layers'],
    dense_units = lstm_model_params['dense_units'],
    input_shape=X_train.shape[1:],
    activations = lstm_model_params['activ'], 
    if_dropout = lstm_model_params['dropout'], 
    dropout_val = lstm_model_params['dropout_val']
)

lstm_model = conf_model(lstm_model_instance, X_train, y_train, X_val, y_val, train_output_path, 
    mtype="DL", epochs=lstm_model_params['epochs'], batch_size=lstm_model_params['batch_size'], B_ARG=B_ARG
)

Y_pred, y_pred_lower, y_pred_upper = lstm_model.predict(X_test, alpha=.1, y_true=y_test, s=None)

lstm_results = evaluate_preds(X_test, y_test, Y_pred, y_pred_lower, y_pred_upper)
plot_predicted_interval(y_test, Y_pred, y_pred_lower, y_pred_upper, "LSTM")
plt.savefig(save_path + dirs[0] + 'lstm_model_with_lags_test_part_cf.png')

Y_train_pred, y_train_pred_lower, y_train_pred_upper = lstm_model.predict(X_train, alpha=.1, y_true=y_train, s=None)
plot_result(y_train, y_test, Y_train_pred, Y_pred,dates = list(dataset.index), title='LSTM')

emissions = tracker.stop()
lstm_results['CO2_EMISSIONS'] = emissions

print("LSTM results: ", lstm_results)



tracker = EmissionsTracker(output_dir=os.path.join(os.getcwd(), save_path, data_params['city']),
    output_file='CO2_detailed_output.csv',
    project_name='GRU-model'
)
tracker.start()

print_banner("Training GRU")

gru_model_params = config['gru_model']
file_name = "GRU_train_output.csv"
train_output_path = os.path.join(save_path, data_params['city'], file_name)

gru_model = flexible_GRU(
    gru_layers=gru_model_params['gru_layers'], hidden_units=gru_model_params['gru_units'], 
    dense_layers=gru_model_params['dense_layers'], dense_units=gru_model_params['dense_units'], 
    input_shape=X_train.shape[1:], activations=gru_model_params['activ'], 
    if_dropout=gru_model_params['dropout'], dropout_val=gru_model_params['dropout_val']
)

GRU = conf_model(gru_model, X_train, y_train, X_val, y_val, mtype = "DL", epochs=gru_model_params['epochs'],
    batch_size=gru_model_params['batch_size'], B_ARG=B_ARG, train_output_path=train_output_path
)
Y_pred, y_pred_lower, y_pred_upper = GRU.predict(X_test, alpha=.1, y_true=y_test, s=None)

gru_results = evaluate_preds(X_test, y_test, Y_pred, y_pred_lower, y_pred_upper)
plot_predicted_interval(y_test, Y_pred, y_pred_lower, y_pred_upper, "GRU")
plt.savefig(save_path + dirs[0] + 'gru_model_with_lags_test_part_cf.png')

Y_train_pred, y_train_pred_lower, y_train_pred_upper = GRU.predict(X_train, alpha=.1, y_true=y_train, s=None)
plot_result(y_train, y_test, Y_train_pred, Y_pred,dates = list(dataset.index), title='GRU')

emissions = tracker.stop()
gru_results['CO2_EMISSIONS'] = emissions

print("GRU results: ", gru_results)



tracker = EmissionsTracker(
    output_dir=os.path.join(os.getcwd(), save_path, data_params['city']),
    output_file='CO2_detailed_output.csv',
    project_name='Conv1d-LSTM'
)
tracker.start()

print_banner("Training Conv1d-LSTM")

conv1d_lstm_model_params = config['conv1d_lstm_model']
file_name = "conv1D_lstm_train_output.csv"
train_output_path = os.path.join(save_path, data_params['city'], file_name)

cnn_lstm_model = Conv1D_LSTM(
    conv_filters=conv1d_lstm_model_params['filters'], conv_kernel_size=conv1d_lstm_model_params['kernel_size'], 
    lstm_layers=conv1d_lstm_model_params['lstm_layers'], lstm_units=conv1d_lstm_model_params['lstm_units'],
    dense_layers=conv1d_lstm_model_params['dense_layers'], dense_units=conv1d_lstm_model_params['dense_units'], 
    input_shape=X_train.shape[1:], activations=conv1d_lstm_model_params['activ'], 
    if_dropout=conv1d_lstm_model_params['dropout'], dropout_val=conv1d_lstm_model_params['dropout_val']
)

CLSTM = conf_model(cnn_lstm_model, X_train, y_train, X_val, y_val, mtype="DL", epochs=conv1d_lstm_model_params['epochs'], 
    batch_size=conv1d_lstm_model_params['batch_size'], B_ARG=B_ARG, train_output_path=train_output_path
)

Y_pred, y_pred_lower, y_pred_upper = CLSTM.predict(X_test, alpha=.1, y_true=y_test, s=None)
conv1D_lstm_results = evaluate_preds(X_test, y_test, Y_pred, y_pred_lower, y_pred_upper)

plot_predicted_interval(y_test, Y_pred, y_pred_lower, y_pred_upper, "CONV1D-LSTM")
plt.savefig(save_path + dirs[0] + 'clstm_model_with_lags_test_part_cf.png')

Y_train_pred, y_train_pred_lower, y_train_pred_upper = CLSTM.predict(X_train, alpha=.1, y_true=y_train, s=None)

plot_result(y_train, y_test, Y_train_pred, Y_pred,dates = list(dataset.index), title='CONV1D-LSTM')

emissions = tracker.stop()
conv1D_lstm_results['CO2_EMISSIONS'] = emissions

print("conv1D_lstm_results: ", conv1D_lstm_results)


"""Comparaison"""
models_results = pd.DataFrame({
    "Linear Regression" : linear_lr_results,
    "Ridge": ridge_results,
    "Random Forest" : rforest_results,
    "XGBoost" : xgboost_results,
    "LightGBM" : lightgbm_results,
    "SVR" : svr_results,
    "Conv1D": conv1D_results,
    "LSTM" : lstm_results,
    "GRU" : gru_results,
    "Conv1D-LSTM": conv1D_lstm_results
}).T

print_banner("Models results")
print(models_results)

models_results.to_excel(save_path + dirs[2]+'Comparaison_ALL_Models.xlsx')
models_results.plot(figsize=(10, 7), kind="bar", rot = 20.0)
plt.savefig(save_path + dirs[0] + 'Comparaison_ALL_Models.png')

borda_count_all = borda_count_ranking(models_results)
borda_count_all.to_excel(save_path + dirs[2]+'BordaCount_ALL_Models.xlsx')

print_banner("Borda Count results")
print(borda_count_all)
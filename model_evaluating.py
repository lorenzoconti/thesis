import pickle
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

from model.gp import gp_predict, gp_create
from model.nn import nn_create, nn_predict

handmade_test_set = False


def df_to_tensor(df, features):
    return torch.from_numpy(np.array(df[features].to_numpy())).float()


df = pd.read_csv('data/learning_dataset.csv', index_col=0)

classes = ['Class_Agg_Urbano', 'Class_Rurale', 'Class_Urbano', 'Class_Montagna']
seasons = ['Spring', 'Summer', 'Fall', 'Winter']
other_categoricals = ['Binary TP', 'Lockdown']

to_be_scaled = ['NH3', 'Altitude', 'Temperature', 'Dewpoint', 'BLH', 'Wind_u', 'Wind_v', 'Wind Speed']

nn_features = to_be_scaled + classes + seasons + other_categoricals
gp_features = ['Longitude', 'Latitude', 'Time']
target = ['NO2']

# remove outliers and use them as test set
outliers_mask = df['Altitude'] > 600
if handmade_test_set:
    # remove the last 27 days as test set
    time_mask = df['Time'] > 1800

scaler = pickle.load(open('output/scaler/scaler.save', 'rb'))
df[to_be_scaled + target] = scaler.fit_transform(df[to_be_scaled + target])

# remove outliers and use them as test set
# test_x_df = df[outliers_mask]

if handmade_test_set:
    # learning data
    df = df[((~outliers_mask) & (~time_mask))][nn_features + gp_features + target]
df = df[~outliers_mask][nn_features + gp_features + target]

train_x_df, test_x_df, train_y_df, test_y_df = train_test_split(df[nn_features + gp_features],
                                                                df[target],
                                                                shuffle=True,
                                                                test_size=0.05,
                                                                random_state=0)

train_x_df, val_x_df, train_y_df, val_y_df = train_test_split(train_x_df,
                                                              train_y_df,
                                                              shuffle=True,
                                                              test_size=0.2,
                                                              random_state=0)

print('train: {}, validation: {}, test: {}'.format(len(train_y_df), len(val_x_df), len(test_x_df)))

# remove the last 27 days as test set
# test_x_df = df[(~outliers_mask) & (time_mask)]
# test_x_df = pd.concat([test_x_df, df[time_mask]], ignore_index=True)

test_x_nn_t = df_to_tensor(test_x_df, nn_features)
test_y_nn_t = df_to_tensor(test_y_df, target)
test_x_gp_t = df_to_tensor(test_x_df, gp_features)

# check device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load nn model
nn_model = nn_create(NUM_FEATURES=len(nn_features), NUM_LAYERS=2, NUM_NEURONS=64)
nn_model.load_state_dict(torch.load('output/models_state_dict/nn_model.pth'))

nn_preds = nn_predict(test_x_nn_t, test_y_nn_t, nn_model, device=device, BATCH_SIZE=1024)

nn_preds = np.array(nn_preds)
test_y_nn = test_y_nn_t.numpy().reshape(-1)

nn_errors = test_y_nn - nn_preds
nn_errors_t = torch.from_numpy(nn_errors)

# load gp model
gp_model, likelihood = gp_create(test_x_gp_t, nn_errors_t, None, NU=1 / 2, INDUCING_POINTS=500)
gp_model.load_state_dict(torch.load('output/models_state_dict/gp_model.pth'))

gp_preds, gp_vars = gp_predict(test_x_gp_t, nn_errors_t, gp_model, 'cuda:0', BATCH_SIZE=128)

gp_preds = np.array(gp_preds)
gp_vars = np.array(gp_vars)

nn_gp_preds = nn_preds + gp_preds
gp_errors = test_y_nn - nn_gp_preds

# plot observed over predicted
num = 150
fig, axs = plt.subplots(nrows=3, figsize=(12, 10))
axs[0].plot(test_y_nn[:num], label='Observed')
axs[0].plot(nn_preds[:num], label='Predicted (NN)', color='black')
axs[0].set_ylabel('Normalized NO₂ (µg/m³)')
axs[0].legend(loc='upper left')

axs[1].plot(test_y_nn[:num], label='Observed', color='grey', alpha=0.5)
axs[1].plot(nn_errors[:num], label='NN Prediction Errors', color='C0', alpha=0.9)
axs[1].plot(gp_preds[:num], label='Predicted (GP)', color='black')
axs[1].set_ylabel('Normalized NO₂ (µg/m³)')
axs[1].legend(loc='upper left')

axs[2].plot(test_y_nn[:num], color='C0', label='Observed')
axs[2].plot(nn_gp_preds[:num], color='black', label='Predicted (NN+GP)')
axs[2].plot(nn_preds[:num], linestyle='dashed', color='grey', label='Predicted (NN)')
axs[2].fill_between(list(range(num)), nn_gp_preds[:num] + 2*np.sqrt(gp_vars[:num]),
                    nn_gp_preds[:num] - 2*np.sqrt(gp_vars[:num]), alpha=0.2, color='grey', label='Uncertainty ±2σ')
axs[2].set_ylabel('Normalized NO₂ (µg/m³)')
axs[2].legend(loc='upper left')

# plt.savefig('output/images/training/ts_predictions_results',  bbox_inches='tight')
plt.show()
#

# plot observed over predicted
fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
axs[0].plot(nn_preds, test_y_nn, '.')
axs[0].set_xlabel('Predicted (NN)')
axs[0].set_ylabel('Observed')

axs[1].plot(nn_gp_preds, test_y_nn, '.')
axs[1].set_xlabel('Predicted (NN+GP)')
axs[1].set_ylabel('Observed')
plt.show()
#

# plot
fig, ax = plt.subplots(figsize=(6, 6))

sns.histplot(nn_errors, ax=ax, color='red', alpha=0.4, label='Prediction errors (NN)')
sns.histplot(gp_errors, ax=ax, color='C0', alpha=0.6, label='Prediction errors (NN+GP)')

ax.vlines(np.median(nn_errors), 0, 190, colors=['red'], linewidth=3, label='Error median (NN)')
ax.vlines(np.median(gp_errors), 0, 190, colors=['blue'], linewidth=3, label='Error median (NN+GP)')

ax.set_xlabel('Normalized NO₂ (µg/m³)')
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
plt.show()
#

# plot observed over predicted
fig, axs = plt.subplots(ncols=2, figsize=(12, 6))

axs[0].plot(nn_gp_preds, test_y_nn, '.')
axs[0].set_xlabel('Predicted (NN+GP)')
axs[0].set_ylabel('Observed')

axs[1].plot(test_y_nn[:100], color='C0', label='Observed')
axs[1].plot(nn_gp_preds[:100], color='black', label='Predicted (NN+GP)')
axs[1].plot(nn_preds[:100], linestyle='dashed', color='grey', label='Predicted (NN)')
axs[1].fill_between(list(range(100)), nn_gp_preds[:100] + 2*np.sqrt(gp_vars[:100]),
                    nn_gp_preds[:100] - 2*np.sqrt(gp_vars[:100]), alpha=0.3, color='grey', label='Uncertainty')
axs[1].set_ylabel('Normalized NO₂ (µg/m³)')
axs[1].legend()
plt.show()
#

print(np.corrcoef(np.array(nn_gp_preds), test_y_nn)[0, 1])
print(r2_score(np.array(nn_preds), test_y_nn))
print(r2_score(np.array(nn_gp_preds), test_y_nn))
print(f'nn+gp rmse: {mean_squared_error(np.array(nn_preds), test_y_nn, squared=True):3f}')
print(f'nn    rmse: {mean_squared_error(np.array(nn_gp_preds), test_y_nn, squared=True):3f}')

performance = pd.DataFrame(data=[
    [
        round(mean_squared_error(np.array(nn_preds), test_y_nn, squared=True), 5),
        round(mean_absolute_error(np.array(nn_preds), test_y_nn), 5),
        round(r2_score(np.array(nn_preds), test_y_nn), 5),

    ],
    [
        round(mean_squared_error(np.array(nn_gp_preds), test_y_nn, squared=True), 5),
        round(mean_absolute_error(np.array(nn_gp_preds), test_y_nn), 5),
        round(r2_score(np.array(nn_gp_preds), test_y_nn), 5),
    ]
],
    columns=['RMSE', 'MAE', 'R2'],
    index=['NN', 'NN+GP'])

print(performance)

performance.to_latex('output/latex/test_performance.tex')

for name, value in gp_model.named_parameters():
    print(name, value)

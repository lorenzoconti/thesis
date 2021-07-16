import torch
import numpy as np
import pandas as pd
import seaborn as sns
import pickle

from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model.dataset import NNGPRegressionDataset
from model.gp import gp_fit, gp_predict
from model.nn import nn_create, nn_fit

# settings
clean = True
handmade_test_set = False

# data
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

# scaling data
scaler = StandardScaler()
scaler = scaler.fit(df[to_be_scaled + target])

df[to_be_scaled + target] = scaler.transform(df[to_be_scaled + target])

if handmade_test_set:
    # learning data
    df = df[((~outliers_mask) & (~time_mask))][nn_features + gp_features + target]
else:
    df = df[~outliers_mask][nn_features + gp_features + target]


# create train and validation data loaders
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


def df_to_tensor(df, features):
    return torch.from_numpy(np.array(df[features].to_numpy())).float()


train_x_t = df_to_tensor(train_x_df, nn_features)
train_y_t = df_to_tensor(train_y_df, target)
train_gp_t = df_to_tensor(train_x_df, gp_features)

train_dataset = NNGPRegressionDataset(x_nn=train_x_t, y_nn=train_y_t, x_gp=train_gp_t)

val_x_t = df_to_tensor(val_x_df, nn_features)
val_y_t = df_to_tensor(val_y_df, target)
val_gp_t = df_to_tensor(val_x_df, gp_features)

val_dataset = NNGPRegressionDataset(x_nn=val_x_t, y_nn=val_y_t, x_gp=val_gp_t)

if clean:
    del train_x_df, val_x_df, train_y_df, val_y_df
    del df

# hyperparameters

# nn
BATCH_SIZE = 256
LEARNING_RATE_NN = 1e-3
NUM_LAYERS = 2
NUM_NEURONS = 64

# gp
INDUCING_POINTS = 500
LEARNING_RATE_GP = 0.02
NU = 1/2

# epochs
EPOCHS_NN = 200
EPOCHS_GP = 150
REPEAT = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# fit neural network
print(f'\n{device} : BATCH_SIZE: {BATCH_SIZE}, LEARNING RATE: {LEARNING_RATE_NN}, '
      f'NUM LAYERS: {NUM_LAYERS}, NUM NEURONS: {NUM_NEURONS}\n')

nn_model = nn_create(NUM_FEATURES=len(nn_features), NUM_LAYERS=NUM_LAYERS,  NUM_NEURONS=NUM_NEURONS)

nn_model, nn_results = nn_fit(model=nn_model,
                              train_dataset=train_dataset,
                              val_dataset=val_dataset,
                              device=device,
                              shuffle=True,
                              adaptive_lr=True,
                              BATCH_SIZE=BATCH_SIZE,
                              LEARNING_RATE=LEARNING_RATE_NN,
                              EPOCHS=EPOCHS_NN,
                              window=25,
                              early_stopping=True,
                              performance_threshold=1e-7)

# smoothing validation error
rmse = nn_results['val_rmse']
smoothed_rmse = []
ms = []
window_size = 25
for i in range(len(rmse)):
    if i > window_size-1:
        smoothed_rmse.append(np.mean(rmse[i:i+window_size]))
        m, b = np.polyfit([x for x in range(i, i+window_size)], rmse[i-window_size:i], deg=1)
        ms.append(m)

smoothed_rmse = [np.nan]*(1 + window_size) + smoothed_rmse

# neural network performance
fig, axs = plt.subplots(ncols=2, figsize=(16, 8))

axs[0].plot(nn_results['train_rmse'], label='Train RMSE')
axs[0].plot(nn_results['val_rmse'], label='Validation RMSE')
axs[0].plot(smoothed_rmse, label='Validation RMSE Moving Average')
axs[0].set_ylabel('RMSE')
axs[0].set_xlabel('Epoch')

axs[1].plot(nn_results['train_r2'], label='Train R²')
axs[1].plot(nn_results['val_r2'], label='Validation R²')
axs[1].set_ylabel('R²')
axs[1].set_xlabel('Epoch')

axs[0].legend()
axs[1].legend()
# plt.savefig('output/images/training/train_metrics_205.png',  bbox_inches='tight')
plt.show()


# predict and get prediction errors
nn_model.to(device)
nn_model.eval()

train_preds = nn_model(train_x_t.to(device))
train_errors = train_y_t.cpu().detach() - train_preds.cpu().detach()

val_preds = nn_model(val_x_t.to(device))
val_errors = val_y_t.cpu().detach() - val_preds.cpu().detach()

# check errors
fig, ax = plt.subplots(figsize=(8, 8))
sns.histplot(train_errors.numpy(), ax=ax, palette=sns.color_palette('coolwarm', n_colors=1), label='Train errors')
sns.histplot(val_errors.numpy(), ax=ax, color='blue', label='Validation errors')
ax.set_xlabel('Normalized NO₂ (µg/m³)')
ax.legend()
plt.show()

# fit gp
print(f'\n{device} : LEARNING RATE: {LEARNING_RATE_GP}, INDUCING POINTS: {INDUCING_POINTS}, NU: {NU}\n')

gp_model, gp_results = gp_fit(
                train_x=train_gp_t,
                train_y=train_errors.squeeze(),
                val_x=val_gp_t,
                val_y=val_errors,
                device=device,
                EPOCHS=EPOCHS_GP,
                LEARNING_RATE=LEARNING_RATE_GP,
                REPEAT=REPEAT,
                INDUCING_POINTS=INDUCING_POINTS,
                NU=NU,
                adaptive_lr=True,
                window=10,
                early_stopping=True,
                performance_threshold=0
            )

# predict and adjust previous predictions
error_preds, error_var = gp_predict(train_gp_t, train_errors.squeeze(), model=gp_model, device=device, BATCH_SIZE=128)

true_values = train_y_t.cpu().detach().numpy().reshape(-1)
predictions = train_preds.cpu().detach().numpy().reshape(-1)
errors = np.array(error_preds).reshape(-1)

predictions_errors = predictions + errors
r2 = r2_score(true_values, predictions_errors)
print(r2)

for (k, v) in gp_model.named_parameters():
    if k != 'covar_module.inducing_points':
        print(f'{k}: {v}')

# plot predictions over true values
fig, ax = plt.subplots(figsize=(8, 8))
m, b = np.polyfit(true_values, predictions_errors, 1)
x = np.arange(np.min(true_values), np.max(true_values))
y = m*x
text = f'R² {round(r2, 3)}'
ax.plot(true_values, predictions, '.', color='lightgrey', label='True values over NN predictions')
ax.plot(true_values, predictions_errors, '.', label='True values over NN+GP predictions')
ax.plot(x, y, color='midnightblue')
ax.set_xlabel('Observed')
ax.set_ylabel('Predicted (NN+GP)')
ax.text(-1.5, 5, text, fontsize=12)
ax.legend(loc='lower right')
plt.show()

# plot predictions over true values with error variances
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(predictions[:100], color='grey', label='Predicted (NN)', linestyle='dashed')
ax.plot(true_values[:100], color='C0', label='Observed')
ax.plot(predictions_errors[:100], color='black', label='Predicted (NN+GP)')

ax.fill_between(list(range(100)), predictions_errors[:100]+2*np.sqrt(error_var[:100]), predictions_errors[:100]-2*np.sqrt(error_var[:100]),
                alpha=0.3, color='gray', label='Uncertainty ±2σ')
ax.set_ylabel('Normalized NO₂ (µg/m³)')
ax.legend()
# plt.savefig('output/images/training/train_error_2sigma.png',  bbox_inches='tight')
plt.show()

#
rmse = nn_results['val_rmse']
smoothed_rmse = []
ms = []
window_size = 10
for i in range(len(rmse)):
    if i > window_size-1:
        smoothed_rmse.append(np.mean(rmse[i:i+window_size]))
        m, b = np.polyfit([x for x in range(i, i+window_size)], rmse[i-window_size:i], deg=1)
        ms.append(m)

smoothed_rmse = [np.nan]*(1 + window_size) + smoothed_rmse

fig, ax = plt.subplots(nrows=1, figsize=(8, 8))
ax.plot(rmse, label='Validation RMSE')
ax.plot(smoothed_rmse, label='Validation RMSE Moving Average')
ax.set_ylabel('RMSE')
ax.set_xlabel('Epoch')

ax.legend()
plt.show()

# save the models
# torch.save(nn_model.state_dict(), 'output/models_state_dict/nn_model_lockdown.pth')
# torch.save(gp_model.state_dict(), 'output/models_state_dict/gp_model_lockdown.pth')

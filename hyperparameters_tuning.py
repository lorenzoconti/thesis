import torch
import numpy as np
import pandas as pd
import pickle

from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model.dataset import NNGPRegressionDataset
from model.gp import gp_fit, gp_predict
from model.nn import nn_create, nn_fit

# settings
clean = True

# data
df = pd.read_csv('data/learning_dataset.csv', index_col=0)

classes = ['Class_Agg_Urbano', 'Class_Rurale', 'Class_Urbano', 'Class_Montagna']
seasons = ['Spring', 'Summer', 'Fall', 'Winter']
other_categoricals = ['Binary TP', 'Weekend']

to_be_scaled = ['NH3', 'Altitude', 'Temperature', 'Dewpoint', 'BLH', 'Wind_u', 'Wind_v', 'Wind Speed']

nn_features = to_be_scaled + classes + seasons + other_categoricals
gp_features = ['Longitude', 'Latitude', 'Time']
target = ['NO2']

# remove outliers and use them as test set
outliers_mask = df['Altitude'] > 600

# remove the last 27 days as test set
time_mask = df['Time'] > 1800

# scaling data
scaler = StandardScaler()
df[to_be_scaled + target] = scaler.fit_transform(df[to_be_scaled + target])

pickle.dump(scaler, open('output/scaler.save', 'wb'))

# learning data
df = df[((~outliers_mask) & (~time_mask))][nn_features + gp_features + target]

# create train and validation data loaders
train_x_df, val_x_df, train_y_df, val_y_df = train_test_split(df[nn_features + gp_features],
                                                              df[target],
                                                              shuffle=True,
                                                              test_size=0.1,
                                                              random_state=0)


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

# neural network hyperparamaters tuning

BATCH_SIZES = [256]
LEARNING_RATES = [1e-3]
NUM_LAYERS = [2]
NUM_NEURONS = [64]

EPOCHS = 150

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
grid_search_df = pd.DataFrame()
grid_search_l = []

previous_epochs = 20
iter_num = 0
max_iter = len(BATCH_SIZES)*len(LEARNING_RATES)*len(NUM_LAYERS)*len(NUM_NEURONS)

for BATCH_SIZE in BATCH_SIZES:
    for LR in LEARNING_RATES:
        for NUM_L in NUM_LAYERS:
            for NUM_N in NUM_NEURONS:

                print(f'\n{device} : {iter_num}/{max_iter} '
                      f'BATCH_SIZE: {BATCH_SIZE}, LEARNING RATE: {LR}, NUM LAYERS: {NUM_L}, NUM NEURONS: {NUM_N}\n')
                iter_num += 1

                nn_model = nn_create(NUM_FEATURES=len(nn_features), NUM_LAYERS=NUM_L,  NUM_NEURONS=NUM_N)

                nn_model, iter_results = nn_fit(model=nn_model,
                                                train_dataset=train_dataset,
                                                val_dataset=val_dataset,
                                                device=device,
                                                BATCH_SIZE=BATCH_SIZE,
                                                LEARNING_RATE=LR,
                                                shuffle=True,
                                                adaptive_lr=False,
                                                EPOCHS=EPOCHS,
                                                early_stopping=False)

                iter_results = pd.DataFrame.from_dict(iter_results)
                grid_search_l.append(iter_results)

                iter_results = iter_results.apply(lambda col: col[-previous_epochs:].mean()).to_frame().T
                iter_results['batch_size'] = BATCH_SIZE
                iter_results['learning_rate'] = LR
                iter_results['num_laters'] = NUM_L
                iter_results['num_neurons'] = NUM_N

                grid_search_df = pd.concat([grid_search_df, iter_results], ignore_index=True)

# grid_search_df.to_csv('output/grid_search_nn_2021_09_05_extended.csv')

# BS 256, LR 1e-3, NL 2, NN 64

# predict values and get errors
nn_model.to(device)
nn_model.eval()

train_preds = nn_model(train_x_t.to(device))
train_errors = train_y_t.cpu().detach() - train_preds.cpu().detach()

val_preds = nn_model(val_x_t.to(device))
val_errors = val_y_t.cpu().detach() - val_preds.cpu().detach()

# gaussian process hyperparameters tuning
INDUCING_POINTS = [150]
LEARNING_RATES = [0.02]
NUS = [1/2]

EPOCHS = 150

iter_num = 0
max_iter = len(NUS)*len(INDUCING_POINTS)*len(LEARNING_RATES)
grid_search_df = pd.DataFrame()

for LR in LEARNING_RATES:
    for IP in INDUCING_POINTS:
        for NU in NUS:

            print(f'\n{device} : {iter_num}/{max_iter} '
                  f'LEARNING RATE: {LR}, INDUCING POINTS: {IP}, NU: {NU}\n')

            iter_num += 1

            gp_model, gp_iter_results = gp_fit(
                train_x=train_gp_t,
                train_y=train_errors.squeeze(),
                val_x=val_gp_t,
                val_y=val_errors,
                device=device,
                EPOCHS=EPOCHS,
                LEARNING_RATE=LR,
                INDUCING_POINTS=IP,
                NU=NU,
                adaptive_lr=False,
            )

            gp_iter_results = pd.DataFrame.from_dict(gp_iter_results)
            grid_search_l.append(gp_iter_results)

            gp_iter_results = gp_iter_results.apply(lambda col: col[-previous_epochs:].mean()).to_frame().T
            gp_iter_results['learning_rate'] = LR
            gp_iter_results['nu'] = NU
            gp_iter_results['incuding_points'] = IP

            grid_search_df = pd.concat([grid_search_df, gp_iter_results], ignore_index=True)

# grid_search_df.to_csv('output/grid_search_gp_02_04.csv')

error_preds, error_var = gp_predict(train_gp_t, train_errors.squeeze(), model=gp_model, device=device, BATCH_SIZE=256)

true_values = train_y_t.cpu().detach().numpy().reshape(-1)
predictions = train_preds.cpu().detach().numpy().reshape(-1)
errors = np.array(error_preds).reshape(-1)

predictions_errors = predictions + errors

# plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(true_values[:5000], predictions_errors[:5000], '.')
plt.show()
#

print(f'train R2: {r2_score(true_values, predictions_errors)}')

# plot
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(true_values[:75], color='black', label='observed')
ax.plot(predictions_errors[:75], color='red', label='nn+gp')
ax.plot(predictions[:75], linestyle='dashed', color='grey', label='nn')
ax.fill_between(list(range(75)), predictions_errors[:75]+np.sqrt(error_var[:75]), predictions_errors[:75]-np.sqrt(error_var[:75]),
                alpha=0.3, color='grey')
ax.legend()
plt.show()
#

# plot
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(true_values[:50], label='observed')
ax.plot(predictions[:50], label='predicted')
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(train_errors[:50], label='observed')
ax.plot(error_preds[:50], label='predicted')
ax.legend()
plt.show()
#

# plot
fig, axs = plt.subplots(ncols=3, figsize=(18, 6))
axs[0].plot(iter_results['train_r2'])
axs[0].plot(iter_results['val_r2'])

axs[1].plot(iter_results['train_loss'])
axs[1].plot(iter_results['val_loss'])

axs[2].plot(iter_results['train_rmse'])
axs[2].plot(iter_results['val_rmse'])

plt.show()
#

print(f'nn+gp rmse: {mean_squared_error(predictions_errors, true_values, squared=True):3f}')
print(f'nn    rmse: {mean_squared_error(predictions, true_values, squared=True):3f}')
print(f'nn+gp   r2: {r2_score(predictions_errors, true_values):3f}')
print(f'nn      r2: {r2_score(predictions, true_values):3f}')


#############################################################################################
#############################################################################################
#############################################################################################

grid_search_df.drop(['train'], inplace=True, axis=1)
grid_search_df.drop(['incuding_points'], inplace=True, axis=1)
grid_search_df.rename(columns={
    'val_loss': 'V Loss',
    'val_r2': 'V R$^2$',
    'val_rmse': 'V RMSE',
    'learning_rate': 'LR',
    'nu': '$nu$'
}, inplace=True)
grid_search_df['V Loss'] = grid_search_df['V Loss'] / 1000


#############################################################################################

t = pd.read_csv('output/hyperparameters_gp,csv', index_col=0)
t.sort_values('V Loss', inplace=True)
t.to_latex('output/latex/gp_gridsearch.tex')
import time

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from torch import nn, optim
from torch.utils.data import DataLoader

from model.dataset import GPRegressionDataset


class NeuralNetworkRegression(nn.Module):

    def __init__(self, input_size, layers_size, num_hidden_layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_size, layers_size)])
        self.layers.extend([nn.Linear(layers_size, layers_size) for _ in range(num_hidden_layers)])
        self.layers.append(nn.Linear(layers_size, 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = self.relu(self.layers[i](x))

        x = self.layers[-1](x)
        return (x)

    def predict(self, x):
        for i in range(len(self.layers) - 1):
            x = self.relu(self.layers[i](x))

        x = self.layers[-1](x)
        return (x)


def nn_create(NUM_FEATURES, NUM_LAYERS, NUM_NEURONS):
    """

    """
    return NeuralNetworkRegression(input_size=NUM_FEATURES, layers_size=NUM_NEURONS, num_hidden_layers=NUM_LAYERS)


def nn_fit(model, device, train_dataset, val_dataset, BATCH_SIZE, LEARNING_RATE, EPOCHS=150, shuffle=False,
           drop_last=True, verbose=False, adaptive_lr=False, early_stopping=False, window=10,
           monitored_performance='val_rmse', performance_threshold=1e-5):
    """

    """
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=drop_last)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=drop_last)

    model.to(device)
    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    stats_dict = {
        'train_loss': [], 'val_loss': [], 'train_r2': [], 'train_mae': [],
        'val_r2': [], 'train_rmse': [], 'val_rmse': [], 'val_mae': []
    }

    smoothed_performance = [np.nan]*window

    if verbose:
        print(f'train: {len(train_loader)*BATCH_SIZE}, val: {len(val_loader)*BATCH_SIZE}')
        print(torch.cuda.get_device_name(torch.cuda.current_device()))

    for e in range(1, EPOCHS + 1):

        start = time.time()

        train_epoch_loss = 0
        val_epoch_loss = 0

        train_true_l = []
        train_preds_l = []

        # training step
        model.train()
        for train_x_batch_nn, train_y_batch_nn, train_x_batch_gp in train_loader:

            train_x_batch_nn, train_y_batch_nn = train_x_batch_nn.to(device), train_y_batch_nn.to(device)
            optimizer.zero_grad()

            train_preds = model(train_x_batch_nn)
            train_loss = loss(train_preds, train_y_batch_nn)

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()

            train_preds_l += list(train_preds.cpu().detach().numpy().reshape(-1))
            train_true_l += list(train_y_batch_nn.cpu().detach().numpy().reshape(-1))

        if adaptive_lr and e > 50:
            for g in optimizer.param_groups:
                g['lr'] = LEARNING_RATE * 0.99

        train_r2 = r2_score(train_true_l, train_preds_l)
        train_rmse = mean_squared_error(train_true_l, train_preds_l, squared=True)
        train_mae = mean_absolute_error(train_true_l, train_preds_l)

        # validation step
        val_preds_l = []
        val_true_l = []
        with torch.no_grad():

            model.eval()
            for val_x_batch_nn, val_y_batch_nn, val_x_batch_gp in val_loader:
                val_x_batch_nn, val_y_batch_nn = val_x_batch_nn.to(device), val_y_batch_nn.to(device)

                val_preds = model(val_x_batch_nn)
                val_loss = loss(val_preds, val_y_batch_nn)
                val_epoch_loss += val_loss.item()

                val_preds_l += list(val_preds.cpu().numpy().reshape(-1))
                val_true_l += list(val_y_batch_nn.cpu().numpy().reshape(-1))

            val_r2 = r2_score(val_true_l, val_preds_l)
            val_rmse = mean_squared_error(val_true_l, val_preds_l, squared=True)
            val_mae = mean_absolute_error(val_true_l, val_preds_l)

        stats_dict['train_loss'].append(train_epoch_loss / len(train_loader))
        stats_dict['val_loss'].append(val_epoch_loss / len(val_loader))

        stats_dict['train_r2'].append(train_r2)
        stats_dict['val_r2'].append(val_r2)

        stats_dict['train_rmse'].append(train_rmse)
        stats_dict['val_rmse'].append(val_rmse)

        stats_dict['train_mae'].append(train_mae)
        stats_dict['val_mae'].append(val_mae)

        print(f'epoch {e + 0:03}: '
              f'TRAIN '
              f'loss: {1000 * (train_epoch_loss / len(train_loader)):.3f}e-03 , '
              f'r2 :  {100*train_r2:.3f} '
              f'rmse : {train_rmse:.3f} '
              f'VAL '
              f'loss: {1000 * (val_epoch_loss / len(val_loader)):.3f}e-03 '
              f'r2 :  {100*val_r2:.3f} '
              f'rmse : {val_rmse:.3f} '
              f'({round(time.time() - start, 2)}s)')

        if early_stopping and e > window:
            smoothed_performance.append(np.mean(stats_dict[monitored_performance][e - window:e]))
            m, _ = np.polyfit([x for x in range(window)], smoothed_performance[e - window:e], deg=1)

            if m > performance_threshold:
                print('Training early stopped at epoch ', e)
                return model, stats_dict

    return model, stats_dict


def nn_predict(x, y, model, device, BATCH_SIZE):

    dataset = GPRegressionDataset(x, y)
    data_loader = DataLoader(dataset, shuffle=False, drop_last=False, batch_size=BATCH_SIZE)

    model.to(device)

    preds_list = []
    with torch.no_grad():
        model.eval()

        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            preds = model(x_batch)
            preds_list += list(preds.cpu().numpy().reshape(-1))

    return preds_list

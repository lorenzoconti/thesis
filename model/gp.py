import gc
import time
import gpytorch
import numpy as np
import torch

from sklearn.metrics import r2_score, mean_squared_error
from torch import optim
from gpytorch.means import ConstantMean
from gpytorch.kernels import InducingPointKernel, MaternKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal
from torch.utils.data import DataLoader

from model.dataset import GPRegressionDataset


class SparseGPRegression(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, NU, INDUCING_POINTS, lengthscales=None):
        super(SparseGPRegression, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.base_covar_module = ScaleKernel(MaternKernel(nu=NU, ard_num_dims=3, eps=1e-6))
        self.covar_module = InducingPointKernel(self.base_covar_module,
                                                inducing_points=train_x[:INDUCING_POINTS, :],
                                                likelihood=likelihood)
        if lengthscales is not None:
            self.covar_module.base_kernel.base_kernel.lengthscale = lengthscales
            self.covar_module.base_kernel.lenghtscale = 1

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


def gp_create(train_x, train_y, lengthscales, NU, INDUCING_POINTS):
    """

    """
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = SparseGPRegression(train_x, train_y, likelihood, NU, INDUCING_POINTS, lengthscales)
    return model, likelihood


def gp_fit(train_x, train_y, val_x, val_y, device, LEARNING_RATE, EPOCHS, REPEAT=1, BATCH_SIZE=1024, NU=1/2,
           INDUCING_POINTS=200, adaptive_lr=False, early_stopping=False, window=10,
           monitored_performance='val_rmse', performance_threshold=1e-5):
    """

    """
    lengthscales = None
    stats_dict = {'train': [], 'val_loss': [], 'val_r2': [], 'val_rmse': []}

    train_x, train_y = train_x.to(device), train_y.to(device)

    val_dataset = GPRegressionDataset(val_x, val_y)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    for r in range(REPEAT):

        model, likelihood = gp_create(train_x, train_y, lengthscales, NU, INDUCING_POINTS)

        model.to(device)
        likelihood = likelihood.to(device)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        smoothed_performance = [np.nan] * window

        for e in range(1, EPOCHS + 1):

            start = time.time()

            train_epoch_loss = 0
            val_epoch_loss = 0
            model.train()

            optimizer.zero_grad()

            if adaptive_lr and e > 100:
                for g in optimizer.param_groups:
                    g['lr'] = LEARNING_RATE*0.99

            with gpytorch.settings.cholesky_jitter(1e-1):
                # why???? TODO
                model.cuda()
                train_preds = model(train_x)
                gp_loss = -mll(train_preds, train_y)

                gp_loss.mean().backward()
                optimizer.step()

                gc.collect()
                torch.cuda.empty_cache()
                train_epoch_loss += gp_loss.item()

            val_preds_list = []
            with torch.no_grad(), gpytorch.settings.fast_pred_var():

                likelihood.eval()
                model.eval()

                for val_x_batch_gp, val_y_batch_gp in val_loader:

                    val_x_batch_gp, val_y_batch_gp = val_x_batch_gp.to(device), val_y_batch_gp.to(device)
                    f_preds = model(val_x_batch_gp)
                    # f_var = f_preds.variance
                    # f_covar = f_preds.covariance_matrix
                    means = f_preds.mean
                    gp_val_loss = -mll(f_preds, val_y_batch_gp)
                    val_epoch_loss += gp_val_loss.mean().item()
                    val_preds_list += [m.squeeze().tolist() for m in means.cpu().detach().numpy()]

                    gc.collect()
                    torch.cuda.empty_cache()

            r2 = r2_score(val_y.cpu().detach().numpy(), val_preds_list)
            rmse = mean_squared_error(val_y.cpu().detach().numpy(), val_preds_list, squared=True)

            stats_dict['train'].append(train_epoch_loss/len(train_x))
            stats_dict['val_loss'].append(val_epoch_loss/len(val_loader))
            stats_dict['val_r2'].append(r2)
            stats_dict['val_rmse'].append(rmse)

            if not e % 10:
                print(f'epoch {e + 0:03}:, '
                      f'VAL '
                      f'| loss: {val_epoch_loss / len(val_loader):.3f} '
                      f'| r2 : {100 * r2:.3f} '
                      f'| rmse : {rmse:.3f} '
                      f'({round(time.time() - start, 2)}s)')

            if early_stopping and e > window:
                smoothed_performance.append(np.mean(stats_dict[monitored_performance][e - window:e]))
                m, _ = np.polyfit([x for x in range(window)], smoothed_performance[e - window:e], deg=1)

                if m > performance_threshold:
                    print('Training early stopped at epoch ', e)
                    return model, stats_dict

            lengthscales = list(model.cpu().named_parameters())[3][1].data.numpy()

    return model, stats_dict


def gp_predict(x, y, model, device, BATCH_SIZE):

    dataset = GPRegressionDataset(x, y)
    data_loader = DataLoader(dataset, shuffle=False, drop_last=False, batch_size=BATCH_SIZE)

    model.to(device)

    preds_list = []
    var_list = []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        model.eval()

        for x_batch, y_batch in data_loader:

            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            f_preds = model(x_batch)
            f_var = f_preds.variance
            # f_covar = f_preds.covariance_matrix
            preds_list += [m.squeeze().tolist() for m in f_preds.mean.cpu().detach().numpy()]
            var_list += [m.squeeze().tolist() for m in f_var.cpu().detach().numpy()]
            torch.cuda.empty_cache()
            gc.collect()

    return preds_list, var_list

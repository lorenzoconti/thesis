from torch.utils.data import Dataset


class NNGPRegressionDataset(Dataset):

    def __init__(self, x_nn, y_nn, x_gp):
        self.x_nn = x_nn
        self.y_nn = y_nn
        self.x_gp = x_gp

    def __getitem__(self, index):
        return self.x_nn[index], self.y_nn[index], self.x_gp[index]

    def __len__(self):
        return len(self.x_nn)

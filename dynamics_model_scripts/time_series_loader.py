import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, X,V):

        self.X_time_series = torch.tensor(X, dtype=torch.float32)
        self.V_time_series = torch.tensor(V,dtype=torch.float32)

    def __len__(self):
        # Length is one less than the number of time steps
        return self.X_time_series.shape[0] - 1

    def __getitem__(self, index):
        # Return the pair (x_n, x_{n+1})
        x_n = self.X_time_series[index]
        x_np1 = self.X_time_series[index + 1]
        v_n = self.V_time_series[index]
        return x_np1,x_n,v_n

class TimeSeriesDataset_w_Prediction(Dataset):
    def __init__(self, X, V, n_step_prediction):
        self.X_time_series = torch.tensor(X, dtype=torch.float32)
        self.V_time_series = torch.tensor(V, dtype=torch.float32)
        self.n_step_prediction = n_step_prediction

    def __len__(self):
        # Length is one less than the number of time steps
        return self.X_time_series.shape[0] - 1 - self.n_step_prediction

    def __getitem__(self, index):
        x_n = self.X_time_series[index]
        v_n = self.V_time_series[index]
        x_future = self.X_time_series[index:index+self.n_step_prediction+1]
        v_future = self.V_time_series[index:index+self.n_step_prediction]
        return x_n, v_n, x_future, v_future
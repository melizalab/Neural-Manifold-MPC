from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import torch

def closed_loop_forecast(ldm_model,X_data,data_loader,device):
    with torch.no_grad():
        for _, (_, X_n, V_n) in enumerate(data_loader):
            # Get true data put on device
            X_n = X_n.to(device)
            V_n = V_n.to(device)
            # Get LDM predictions
            X_np1_hat, U_n_hat, Z_np1, Z_n, _, _= ldm_model(X_n,V_n)
            # Last layer is a bernoulli
            #X_np1_hat = torch.bernoulli(X_np1_hat)
    return X_np1_hat.detach().cpu().numpy(),Z_np1.detach().cpu().numpy(),Z_n.detach().cpu().numpy()


def open_loop_forecast(ldm_model,X_data,V_data,device,n_obs):
    X_np1_hat = torch.zeros((n_obs, 122)).to(device)
    Z_np1_hat = torch.zeros((n_obs, 2)).to(device)
    # Covert V input numpy array to torch tensor
    V_data = torch.from_numpy(V_data).to(torch.float32).to(device)
    # Seed initial value of X state prediction
    X_np1_hat[0] = torch.from_numpy(X_data[1,:]).to(torch.float32).to(device)
    with torch.no_grad():
        for i in range(1, n_obs):
            X_np1_hat[i],_,Z_np1_hat[i],_,_,_ = ldm_model(X_np1_hat[i - 1].view(1, -1), V_data[i - 1].view(1, -1))
            #X_np1_hat[i] = torch.bernoulli(x_np1_hat)
    return X_np1_hat.detach().cpu().numpy(),Z_np1_hat.detach().cpu().numpy()
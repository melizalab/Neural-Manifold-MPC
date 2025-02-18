import numpy as np

def linear_interpolation(A,B,i,len_of_interp):
    return A*(1-i/len_of_interp)+B*(i/len_of_interp)

def interpolate_sequence(data,len_of_interp,params):
    n_time_points = (data.shape[0]-1)*len_of_interp
    interpolated_X = np.zeros((n_time_points,params['latent_dim_size']))
    counter = 0
    for i in range(n_time_points):
        if i>0 and i%len_of_interp==0:
            counter+=1
        interpolated_X[i] = linear_interpolation(data[counter],data[counter+1],i%len_of_interp,len_of_interp)

    return interpolated_X,n_time_points

def get_latent_trajectory(centers,train_val_test,step_len,slow_len,fast_len,params):
    centers_1 = centers[:len(centers)//2, :]
    centers_2 = centers[len(centers)//2:, :]
    # Lengths for step are how long each image is static before jumping
    step_trajectory = np.repeat(centers_1, step_len, axis=0)
    # Lengths should be shorter for fast traj since its interp time not full stim time
    slow_trajectory, _ = interpolate_sequence(centers_2, slow_len,params)
    fast_trajectory, _ = interpolate_sequence(centers, fast_len,params)
    if train_val_test=='train':
        return np.concatenate((step_trajectory,slow_trajectory,fast_trajectory))
    if train_val_test=='val':
        return np.concatenate((fast_trajectory,step_trajectory,slow_trajectory))
    elif train_val_test=='test':
        return np.concatenate((slow_trajectory, fast_trajectory,step_trajectory))
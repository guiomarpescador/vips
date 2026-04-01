import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import gpflow
import tensorflow as tf
from src.models.osgpr import OSGPR_VFE
from gpflow.models import SGPR

from utils_magnetometer import convert_data_to_online, load_data, transform_grid2room, transform_room2video, get_transformed_grid

"""
Plot predictions function is based on https://github.com/AaltoML/sequential-gp/blob/main/experiments/magnetometer/online_model_predictions.py
"""

# Times new roman font
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
# Bolds font
plt.rcParams.update({'font.size': 20})
# legend fontsize
plt.rcParams["legend.fontsize"] = 14



# Function to load model parameters from a pickle file
def load_model_parameters(data, filename):
    with open(filename, "rb") as f:
        loaded_params = pickle.load(f)

    # Assign parameters to variables
    kernel = gpflow.kernels.Sum([gpflow.kernels.Constant(), gpflow.kernels.Matern52()])
    kernel.kernels[0].variance.assign(loaded_params['.kernel.kernels[0].variance'])
    kernel.kernels[1].lengthscales.assign(loaded_params['.kernel.kernels[1].lengthscales'])
    kernel.kernels[1].variance.assign(loaded_params['.kernel.kernels[1].variance'])

    Z = loaded_params['.inducing_variable.Z']
    # If file name contains CV_EG, then we need to load the old parameters
    if "0" in filename:
        model = SGPR(data, kernel=kernel, inducing_variable=Z)
    else:
        Z_old = loaded_params['.Z_old']
        Kaa_old = loaded_params['.Kaa_old']
        mu_old = loaded_params['.mu_old']
        Su_old = loaded_params['.Su_old']
        model = OSGPR_VFE(data, kernel=kernel, mu_old=mu_old, Su_old=Su_old, Kaa_old=Kaa_old, Z_old=Z_old, Z=Z)

    model.likelihood.variance.assign(loaded_params['.likelihood.variance'])

    return model

def plot_predictions(methods):
    
    # Get the data
    train_data, _ = load_data('experiments/magnetometer/data/invensense/', train_id=[3])
    online_data = convert_data_to_online(train_data[0], 20, shuffle=False)
    output_dir = "experiments/magnetometer/results/figures/"
    input_dir = "experiments/magnetometer/results/models/"

    batches = [0, 4, 9, 14, 19]

    for method in methods:

        path_x = None

        for i in batches:
        
            # Load the model parameters
            model_name = method + "_model_path_3_batch_" + str(i)
            param_filename = os.path.join(input_dir, model_name + "_params.pkl")

            if path_x is None:
                path_x = online_data[i][0]
            else: 
                path_x = np.concatenate([online_data[j][0] for j in range(i+1)], axis=0)
            
            model = load_model_parameters(online_data[i], param_filename)
            
            print(f"{model_name} loaded successfully!!!")

            # Prediction over grid
            xtest, ytest = np.mgrid[-1:1:100j, -1:1:100j]
            xtest_transformed, ytest_transformed = transform_grid2room(xtest, ytest)
            zz = np.concatenate([xtest_transformed[..., None], ytest_transformed[..., None]], axis=1)

            pred_m_grid, pred_S_grid = model.predict_f(zz)
            pred_m_grid = pred_m_grid.numpy().reshape((100, -1))

            pred_S_grid = pred_S_grid.numpy()
            alpha_map = np.power(pred_S_grid, 0.3).reshape((100, 100)) 
            alpha_map = alpha_map - np.min(alpha_map)
            alpha_map = alpha_map/np.max(alpha_map)
            alpha_map = 1 - alpha_map


            # Test points
            transformed_x1test, transformed_x2test = transform_room2video(xtest_transformed, ytest_transformed)
            transformed_x1test = np.reshape(transformed_x1test, xtest.shape)
            transformed_x2test = np.reshape(transformed_x2test, ytest.shape)

            # Path
            path_transformed_x0, path_transformed_x1 = transform_room2video(path_x[:, 0], path_x[:, 1])
            path_transformed_x = np.concatenate([path_transformed_x0[..., None], path_transformed_x1[..., None]], axis=1)

            # Grid
            g1, g2 = get_transformed_grid()

            # Inducing variables
            Z_new = model.inducing_variable.Z.numpy()
            transformed_Z_0, transformed_Z_1 = transform_room2video(Z_new[:, 0], Z_new[:, 1])
            transformed_Z_0 = np.reshape(transformed_Z_0, Z_new[:, 0].shape)
            transformed_Z_1 = np.reshape(transformed_Z_1, Z_new[:, 1].shape)

            # Plotting
            plt.clf()
            fig, axs = plt.subplots(figsize=(5, 3))  # control dimensions
            fig.patch.set_facecolor('white')
            axs.set_facecolor('white')
            plt.plot(path_transformed_x[:, 0], path_transformed_x[:, 1], lw = 1)
            pcol = plt.pcolormesh(transformed_x1test, transformed_x2test, pred_m_grid, alpha=alpha_map.reshape(-1), shading='gouraud', cmap="jet", 
                                vmin=10, vmax=90)
            pcol.set_edgecolor('face')

            plt.scatter(transformed_Z_0, transformed_Z_1, color="black", s=5)

            plt.plot(g1, g2, color="gray", alpha=0.2)
            plt.plot(g1.T, g2.T, color="gray", alpha=0.2)

            # RMSE = {'VIPS': 7.58, 'CV': 7.85}
           # plt.annotate(f"M = {model.inducing_variable.num_inducing}", xy=(0.8, 0.18), xycoords='axes fraction', ha='center', va='center')
            # if i == 19:
            #     plt.annotate(f"RMSE: {RMSE[method]}", xy=(0.8, 0.08), xycoords='axes fraction', ha='center', va='center')
            
            plt.xlim([0, 1920])
            plt.ylim([0, 1080])
            axs.set_aspect("equal")
            plt.axis('off')
            plt.gca().invert_yaxis()
            # Save as pdf
            plt.savefig(os.path.join(output_dir, model_name + ".png"), bbox_inches='tight', dpi=300)
            plt.savefig(os.path.join(output_dir, model_name + ".pdf"), bbox_inches='tight')
            

if __name__ == "__main__":
    methods = ['VIPS']
    plot_predictions(methods)
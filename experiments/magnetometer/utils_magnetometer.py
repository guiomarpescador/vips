import csv
import os
import gpflow
import pickle

import numpy as np
import pandas as pd

"""
Code for magnetometer data and plotting from: https://github.com/AaltoML/sequential-gp/tree/main/experiments/magnetometer
"""

A_room2video = np.array([70.7791, -388.6396, 618.0954,
                         -66.1362, 26.6245, 665.3620,
                         0, 0, 1.0000]).reshape((3, 3))

C_room2video = np.array([0.1597, -0.0318, 1.0000]).reshape((1, 3))

A_grid2room = np.array([0.0118, 2.7777, 1.3689,
                        -2.2243, -0.0967, -1.0929,
                        0, 0, 1.0000]).reshape((3, 3))

C_grid2room = np.array([0.0160, -0.1450, 1.0000]).reshape((1, 3))


output_dir = "experiments/magnetometer/results/"


def load_data(main_dir: str, train_id: list = None, test_id: list = None) -> [list, list]:
    """
    Load magnetometer data.

    Main source of data is: https://github.com/AaltoML/magnetic-data

    Note: The function involves some constants that are specific to the data.
    """
    data_train = []
    data_test = None

    if train_id is None:
        train_id = [1, 2, 3, 4, 5]

    if test_id is None:
        test_id = [1]

    for i in train_id:
        loc_path = os.path.join(main_dir, str(i) + "-loc.csv")
        mag_path = os.path.join(main_dir, str(i) + "-mag.csv")

        loc_data = pd.read_csv(loc_path).to_numpy()
        mag_data = pd.read_csv(mag_path).to_numpy()

        # take norm of mag data
        mag_data_norm = np.sqrt(np.sum(np.square(mag_data), axis=-1))[..., None]
        data_combined = np.concatenate([loc_data, mag_data_norm], axis=1)

        data_train.append([data_combined[:, :-1], data_combined[:, -1:]])

    for i in test_id:
        loc_path = os.path.join(main_dir, str(i) + "-loc.csv")
        mag_path = os.path.join(main_dir, str(i) + "-mag.csv")

        loc_data = pd.read_csv(loc_path).to_numpy()
        mag_data = pd.read_csv(mag_path).to_numpy()

        # take norm of mag data
        mag_data_norm = np.sqrt(np.sum(np.square(mag_data), axis=-1))[..., None]

        data_combined = np.concatenate([loc_data, mag_data_norm], axis=1)

        if data_test is None:
            data_test = [np.array(data_combined[:, :-1]), np.array(data_combined[:, -1:])]
        else:
            data_test = [np.concatenate([data_test[0], np.array(data_combined[:, :-1])], axis=0),
                         np.concatenate([data_test[1], np.array(data_combined[:, -1:])], axis=0)]

    return data_train, data_test

def convert_data_to_online(data: [np.ndarray, np.ndarray], n_sets: int,
                           shuffle: bool = False, sort_data: bool = False) -> list:
    """
    Get an offline data and convert it into an online dataset of n_sets.

    returns: a list of tuple of np.ndarray (X_i, Y_i) with X_i of shape [n_set_data, data_dim] and
             Y is of shape [n_set_data, output_dim].
    """
    Y_dtype = data[1].dtype

    X, Y = data
    XY = np.concatenate([X, Y], axis=1)

    if shuffle:
        np.random.shuffle(XY)

    if sort_data:
        np.sort(XY, axis=0)

    n = XY.shape[0]
    last_set_size = int(n % n_sets)
    set_size = int((n - last_set_size) / n_sets - 1)

    streaming_data = []
    for i in range(n_sets - 1):
        set_data = XY[i * set_size: (i + 1) * set_size]
        Y_casted = set_data[:, X.shape[-1]:].astype(Y_dtype)
        streaming_data.append((set_data[:, :X.shape[-1]], Y_casted))

    # Adding last set; this could be more than other sets as well
    set_data = XY[(n_sets - 1) * set_size:]

    Y_casted = set_data[:, X.shape[-1]:].astype(Y_dtype)
    streaming_data.append((set_data[:, :X.shape[-1]], Y_casted))

    assert len(streaming_data) == n_sets
    assert streaming_data[0][0].shape[-1] == X.shape[-1]
    assert streaming_data[0][1].shape[-1] == Y.shape[-1]

    return streaming_data

"""
Below function are for plotting purposes and comes from original Matlab scripts.

They are for transformation between room, video, grid.

Again from: https://github.com/AaltoML/sequential-gp/tree/main/experiments/magnetometer
"""

def get_transformed_grid():
    z1 = np.concatenate([np.linspace(-1, 1, 32), np.nan * np.ones((1,))])
    z2 = z1.copy()

    g1, g2 = np.meshgrid(z1, z2)
    Z = np.concatenate([g1.reshape((-1, 1)), g2.reshape((-1, 1))], axis=1)

    var1 = A_grid2room @ np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=-1).T
    var2 = C_grid2room @ np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=-1).T

    Z = np.divide(var1, var2).T
    Z = Z[:, :2]

    var1 = A_room2video @ np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=-1).T
    var2 = C_room2video @ np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=-1).T
    Y = np.divide(var1, var2).T

    g1 = np.reshape(Y[:, 0], g1.shape)
    g2 = np.reshape(Y[:, 1], g2.shape)

    return g1, g2


def transform_grid2video(x, y):
    Z = np.concatenate([x.reshape((-1, 1)), y.reshape((-1, 1))], axis=1)
    var1 = A_grid2room @ np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=-1).T
    var2 = C_grid2room @ np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=-1).T

    Z = np.divide(var1, var2).T
    Z = Z[:, :2]

    var1 = A_room2video @ np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=-1).T
    var2 = C_room2video @ np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=-1).T
    Y = np.divide(var1, var2).T
    return Y[:, 0], Y[:, 1]


def transform_room2video(x, y):
    Z = np.concatenate([x.reshape((-1, 1)), y.reshape((-1, 1))], axis=1)
    var1 = A_room2video @ np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=-1).T
    var2 = C_room2video @ np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=-1).T
    Y = np.divide(var1, var2).T

    return Y[:, 0], Y[:, 1]


def transform_grid2room(x, y):
    Z = np.concatenate([x.reshape((-1, 1)), y.reshape((-1, 1))], axis=1)
    var1 = A_grid2room @ np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=-1).T
    var2 = C_grid2room @ np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=-1).T
    Y = np.divide(var1, var2).T
    return Y[:, 0], Y[:, 1]


"""
Utility functions for saving models and metrics.
"""

def save_model(model, model_name):
    param_dict = gpflow.utilities.parameter_dict(model)
    with open(os.path.join(output_dir, "models/" + model_name + "_params.pkl"), "wb") as f:
        pickle.dump(param_dict, f)

def save_metrics(metrics, method):
    file_path = os.path.join(output_dir, "metrics/metrics_" + method + ".csv")
    
    # Check if file exists and is not empty
    file_exists = os.path.isfile(file_path) and os.path.getsize(file_path) > 0

    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f)

        # Convert arrays to string
        metrics['rmse'] = ','.join(map(str, metrics['rmse']))
        metrics['nlpd'] = ','.join(map(str, metrics['nlpd']))

        # Write header only if the file did not exist or was empty
        if not file_exists:
            writer.writerow(metrics.keys())

        writer.writerow(metrics.values())
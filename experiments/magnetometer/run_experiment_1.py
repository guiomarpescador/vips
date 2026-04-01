import numpy as np

from utils_regression import run_regression_streaming
from utils_magnetometer import load_data, convert_data_to_online, save_metrics

def run_experiment():
    ### Load data
    data_train, data_test = load_data('experiments/magnetometer/data/invensense/', train_id=[3], test_id=[1,2,4,5] )
    ### Convert data to online
    no_batches = 20
    np.random.seed(0)
    online_data = convert_data_to_online(data_train[0], no_batches, shuffle=False)

    methods = ['OIPS',  'VIPS']
    methods_params = {'M' : 5000, 'eta': 0.06, 'delta': 0.095, 'rho': 0.93}
    index = [0, 4, 9, 14, 19]
    for method in methods:
        np.random.seed(42)
        rmse, nlpd, model, _ = run_regression_streaming(online_data, data_test, method, save_batch = True, **methods_params)
        metrics = {'model': 'path 3', 'rmse': np.round(rmse[index], 3), 'nlpd': np.round(nlpd[index], 3)}
        save_metrics(metrics, method)

if __name__ == "__main__":
    run_experiment()
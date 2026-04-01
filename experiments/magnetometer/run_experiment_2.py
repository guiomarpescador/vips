import numpy as np

from utils_regression import run_regression_sequential, run_regression_streaming
from utils_magnetometer import load_data, convert_data_to_online, save_metrics, save_model

    
def run_experiment():
    ### Load data
    methods = ['OIPS', 'VIPS']
    methods_params = {'M' : 5000, 'eta': 0.06, 'delta': 0.095, 'rho': 0.93}
    paths = [1, 2, 4, 5]
    index = [0]
    no_batches = 1
    np.random.seed(0)
    all_train_data, data_test = load_data('experiments/magnetometer/data/invensense/', train_id=paths, test_id=[3])

    for method in methods:
        for i, train_data in enumerate(all_train_data):
            print(f"Path {i+1} of {len(all_train_data)}...")
            online_data = convert_data_to_online((train_data[0], train_data[1]), no_batches, shuffle=False)
            if i == 0:
                rmse, nlpd, model, params = run_regression_streaming(online_data, (train_data[0], train_data[1]), method, **methods_params)
            else:
                rmse, nlpd, model = run_regression_sequential(online_data, (train_data[0], train_data[1]), method, model, params, **methods_params)

            save_model(model, method + "_method_model_path_" + str(paths[i]))
        
            metrics = {'model': f'path {paths[i]}', 'rmse': np.round(rmse[index],2), 'nlpd': np.round(nlpd[index],2)}
            save_metrics(metrics, method)
                
            print(f"Model {i+1} metrics: RMSE: {np.round(rmse[-1],2)}, NLPD: {np.round(nlpd[-1],2)}")
            print(f"Model {i+1}: saved successfully!")


if __name__ == "__main__":
    run_experiment()
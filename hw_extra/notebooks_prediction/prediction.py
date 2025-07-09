import tensorflow as tf
import os
import numpy as np
import random
import keras
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout, Conv1D, Flatten, Reshape

from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb
import sys
import time

SEED = 42

def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)

def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

# Call the above function with seed value
set_global_determinism(seed=SEED)

class XGBCustomObjective(BaseEstimator, RegressorMixin):
    """
    Wrapper class to make XGBoost with custom objectives compatible with sklearn.
    This is needed for MultiOutputRegressor.
    """
    def __init__(self, objective_func=None, random_state=42, n_estimators=15, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8, **kwargs):
        self.objective_func = objective_func
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.kwargs = kwargs
        self.model = None
        
    def fit(self, X, y):
        """Fit the model with custom objective."""
        # Create DMatrix
        dtrain = xgb.DMatrix(X, label=y)
        
        # Set up parameters using your specified values
        params = {
            'max_depth': self.max_depth,
            'eta': self.learning_rate,  # learning_rate
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'seed': self.random_state,  # random_state
            'disable_default_eval_metric': 1 if self.objective_func else 0
        }
        params.update(self.kwargs)
        
        if self.objective_func:
            # Train with custom objective
            self.model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=self.n_estimators,
                obj=self.objective_func,
                verbose_eval=False
            )
        else:
            # Train with standard objective
            params['objective'] = 'reg:squarederror'
            del params['disable_default_eval_metric']
            self.model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=self.n_estimators,
                verbose_eval=False
            )
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator (required for sklearn compatibility)."""
        params = {
            'objective_func': self.objective_func,
            'random_state': self.random_state,
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree
        }
        params.update(self.kwargs)
        return params
    
    def set_params(self, **params):
        """Set parameters for this estimator (required for sklearn compatibility)."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
        return self

os.chdir("../")
# change working directory to project's root path
print(os.getcwd())

FIRST_YEAR= 1972
FREQUENCY= "monthly"
folder_path = os.path.abspath("functions/") #INPUT_PATH)#'path_to_your_folder')  # Replace with the actual folder path
sys.path.insert(0, folder_path)

from Predictions import (
    PredictionExperiment,
    PredictionModel,
    SERA,
    sera_objective,
    piecewise_linear_phi_np
)

############ Execution #####################

indices_of_interest = ["HWN", "HWF", "HWD", "HWM", "HWA"]
bounds_v1 = (-1.1692892810242344, -0.30647585455315646, 4.561547586528888, 6.499969486244418)
i_w_v1 = 0.1
# bounds = (-1.1692892810242344, -0.30647585455315646, 3.0, 6.499969486244418)
# i_w = 0.3
region="chile"
metadata_path = f"data/climate_features/{region}/metadata.csv"
metadata = pd.read_csv(metadata_path)
metadata.reset_index(inplace=True, drop=True)

results = pd.read_csv(f"data/sera_results_v2/{region}_results/results.csv")
ids_results = results["id_data"].unique()
id_experiments = metadata["id"].unique()
ids_to_execute = [id for id in id_experiments if id not in ids_results]
# print(id_experiments[10:])
k=0
for id in id_experiments[-2:]:
    t1 = time.time()
    k+=1
    print("Executing",id, "iter", k)
    data = {i: pd.read_parquet(f"data/climate_features/{region}/predictor_{id}_{i}.parquet") for i in range(1,13)}
    rnn16_model = Sequential([
    SimpleRNN(16, activation="tanh", input_shape=(1, len(data[1].columns) - len(indices_of_interest))),
    Dropout(0.1),  # Regularization
    Dense(8, activation="relu"),
    Dense(len(indices_of_interest))  # Predict 5 indices
    ])
    lstm16_model = Sequential([
    LSTM(16, activation="tanh", input_shape=(1, len(data[1].columns) - len(indices_of_interest))),
    Dropout(0.1),  # Regularization
    Dense(8, activation="relu"),
    Dense(len(indices_of_interest))  # Predict 5 indices
    ])
    cnn_rnn_model = Sequential([
        Conv1D(16, kernel_size=1, activation="relu", input_shape=(1, len(data[1].columns) - len(indices_of_interest))),
        Reshape((1, 16)),  # Back to time dimension
        SimpleRNN(8, activation="tanh"),
        Dropout(0.1),
        Dense(len(indices_of_interest))
    ])
    lp_model = Sequential([
        Flatten(input_shape=(1, len(data[1].columns) - len(indices_of_interest))),
        Dense(16, activation="relu"),
        Dropout(0.1),
        Dense(8, activation="relu"),
        Dense(len(indices_of_interest))
    ])
    xgb_model = XGBCustomObjective(
        objective_func=sera_objective(piecewise_linear_phi_np(bounds_v1, initial_weight=i_w_v1)),
        n_estimators=15,
        learning_rate=0.1
    )
    # assert len(regressors) == len(name_regressors)
    if id in ['978f49d7', '69ae08a8', '1b939ac5']:
        regressors = [xgb_model]
        name_regressors = ["CXGB15"]
    else:
        regressors =  [xgb_model, rnn16_model, lstm16_model, cnn_rnn_model, lp_model]
        name_regressors =  ["CXGB15", "RNN16", "LSTM16", "CNNRNN16", "MLP16"]
    assert len(regressors) == len(name_regressors)
    experiment_1 = PredictionExperiment(data, indices_of_interest, regressors, name_regressors, 5, id, loss_fn=SERA(bounds=bounds_v1,T=100, initial_weight=i_w_v1, fn="piecewise2"))
    experiment_1.execute_experiment()
    experiment_1.get_metrics("r2", "prediction", show=False)
    experiment_1.get_metrics("mape", "prediction", show=False)
    experiment_1.get_metrics("mae", stage="prediction", show=False)
    experiment_1.get_metrics("r2", stage="training", show=False)
    experiment_1.get_metrics("mape", stage="training", show=False)
    experiment_1.get_metrics("mae", stage="training", show=False)
    experiment_1.get_metrics("r2", stage="CV", show=False)
    experiment_1.get_metrics("mape", stage="CV", show=False)
    experiment_1.get_metrics("mae", stage="CV", show=False)
    experiment_1.get_metrics("r2", stage="TSCV", show=False)
    experiment_1.get_metrics("mape", stage="TSCV", show=False)
    experiment_1.get_metrics("mae", stage="TSCV", show=False)
    experiment_1.get_metrics("sera", stage="prediction", show=False)
    experiment_1.get_metrics("sera", stage="training", show=False)
    experiment_1.get_metrics("sera", stage="CV", show=False)
    experiment_1.get_metrics("sera", stage="TSCV", show=False)

    #experiment_1.top_results("r2", 5, stage="prediction", top_data_path=f"data/results/{FREQUENCY}/top_results.csv")
    #experiment_1.top_results("cv_r2", 5, stage="CV", top_data_path=f"data/results/{FREQUENCY}/top_results.csv")
    experiment_1.save_results(f"data/sera_results_v2/{region}_results/results.csv")
    del experiment_1
    print("Time experiment", time.time()-t1)
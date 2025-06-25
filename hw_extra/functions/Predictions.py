import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.inspection import permutation_importance

from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.multioutput import MultiOutputRegressor
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from tensorflow_addons.metrics import RSquare

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from xgboost import plot_importance

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
import tensorflow as tf
import shap
tf.random.set_seed(42)
import keras
import copy
from sklearn.model_selection import KFold
import tensorflow.keras.backend as K

from scipy.interpolate import PchipInterpolator
from statsmodels.stats.stattools import medcouple

simplefilter("ignore", category=ConvergenceWarning)


### UTILS

def get_info_experiment(id_data, metadata_exp_path, metadata_index_path, extra_indices_path):
    metadata = pd.read_csv(metadata_exp_path)
    metadata_indices = pd.read_csv(metadata_index_path)
    metadata_indices.set_index("id",inplace=True)
    metadata_extra = pd.read_csv(extra_indices_path)
    metadata_extra.set_index("id",inplace=True)

    ids_indices = metadata[metadata["id"]==id_data]["indices"].unique()[0].split("-")
    my_indices = [id for id in ids_indices if len(id)==8]
    extra_indices = [id for id in ids_indices if len(id)!=8]
    
    return pd.concat([metadata_indices.loc[my_indices],metadata_extra.loc[extra_indices]], axis=0)

def summarize_best_results_by_index(results, metadata, metric="r2", stage="prediction", top_n=1, exclude_model="GPR-rbf-noise", indices_of_interest=["HWN", "HWF", "HWD", "HWM", "HWA", "Average"]):

    # Filter for prediction stage based on metric
    # prediction_results = results[(results["stage"] == stage) & (results["metric"] == metric) & (results["model"]!= "Linear")]
    prediction_results = results[(results["stage"] == stage) & (results["metric"] == metric)]


    # Find the top N best values per index (maximize for r2 and cv_r2, minimize for mape)
    best_results = prediction_results.set_index(["model", "season", "id_data"])[indices_of_interest].stack().reset_index()
    best_results.columns = ["model", "season", "id_data", "index", "best_value"]
    
    if metric =="r2":
        best_results = best_results.groupby("index").apply(lambda x: x.nlargest(top_n, "best_value")).reset_index(drop=True)
    else:  # metric == "mape"
        best_results = best_results.groupby("index").apply(lambda x: x.nsmallest(top_n, "best_value")).reset_index(drop=True)

    # Get corresponding training values
    if metric in ["r2", "mape", "mae", "sera"] and stage not in ["CV","TSCV"]:
        training_results = results[(results["stage"] == "training") & (results["metric"] == metric) & results["id_data"].isin(best_results["id_data"])]
        training_results = training_results.set_index(["model", "season", "id_data"])[indices_of_interest].stack().reset_index()
        training_results.columns = ["model", "season", "id_data", "index", "training_value"]

        # Merge best prediction results with training values
        summary = best_results.merge(training_results, on=["model", "season", "id_data", "index"], how="left")

    else:
        summary = best_results
    # Merge with metadata
    summary = summary.merge(metadata, on=["id_data","season"], how="left")

    # Save summary
    #summary.to_csv(f"summary_best_{metric}_results.csv", index=False)

    return summary

def summarize_all_results(results, metadata, metric="r2", stage="prediction", top_n=1, exclude_model="GPR-rbf-noise", indices_of_interest=["HWN", "HWF", "HWD", "HWM", "HWA", "Average"]):

    # Filter for prediction stage based on metric
    # prediction_results = results[(results["stage"] == stage) & (results["metric"] == metric) & (results["model"]!= "Linear")]
    prediction_results = results[(results["stage"] == stage) & (results["metric"] == metric) & (results["id_data"]!="c6b5cfea")]


    # Find the top N best values per index (maximize for r2 and cv_r2, minimize for mape)
    best_results = prediction_results.set_index(["model", "season", "id_data", "source"])[indices_of_interest].stack().reset_index()
    best_results.columns = ["model", "season", "id_data", "source", "index", "best_value"]
    
    if metric =="r2":
        best_results = best_results.groupby("index").apply(lambda x: x.nlargest(top_n, "best_value")).reset_index(drop=True)
    else:  # metric == "mape"
        best_results = best_results.groupby("index").apply(lambda x: x.nsmallest(top_n, "best_value")).reset_index(drop=True)

    # Get corresponding training values
    if metric in ["r2", "mape", "mae"] and stage not in ["CV","TSCV"]:
        training_results = results[(results["stage"] == "training") & (results["metric"] == metric) & results["id_data"].isin(best_results["id_data"])]
        training_results = training_results.set_index(["model", "season", "id_data", "source"])[indices_of_interest].stack().reset_index()
        training_results.columns = ["model", "season", "id_data","source", "index", "training_value"] 

        # Merge best prediction results with training values
        summary = best_results.merge(training_results, on=["model", "season", "id_data", "source", "index"], how="left")

    else:
        summary = best_results
    # Merge with metadata
    summary = summary.merge(metadata, on=["id_data","season", "source"], how="left")
    summary.drop(columns="filename", inplace=True)
    # Save summary
    #summary.to_csv(f"summary_best_{metric}_results.csv", index=False)

    return summary

def plot_best_results_per_season(df, metric, stage, title):
    seasons = sorted(df["season"].unique())
    
    fig, axes = plt.subplots(3, 5, figsize=(18, 10))  # Adjust layout
    axes = axes.flatten()

    # Get unique models for the legend
    unique_models = df["model"].unique()
    palette = sns.color_palette("tab10", len(unique_models))  # Assign colors
    model_colors = dict(zip(unique_models, palette))  # Map models to colors

    for i, season in enumerate(seasons):
        ax = axes[i]
        season_data = df[df["season"] == season]

        sns.barplot(
            data=season_data, 
            x="index", y="best_value", hue="model", ax=ax, 
            dodge=True, palette=model_colors
        )
        if season==0:
            ax.set_title(f"Yearly")
        else:
            ax.set_title(f"Season {season}")
        ax.set_xlabel("Index")
        ax.set_ylabel(metric)
        ax.legend().remove()  # Remove subplot legends
        if "MAPE" in metric and "CV"==stage:
            ax.set_ylim(bottom=-1, top=0)
        else:
            ax.set_ylim(bottom=0, top=1)
    
    # Create a single global legend
    handles = [plt.Rectangle((0,0),1,1, color=model_colors[model]) for model in unique_models]
    fig.legend(handles, unique_models, title="Model", loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=len(unique_models))

    # Adjust layout
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.90)  # Space for the suptitle and legend
    plt.show()

def plot_average_best_results(df, metric, title):
    # Compute the average best result per index across all seasons
    df1 = df[df["season"]!=0]
    avg_best_results = df1.groupby("index")["best_value"].mean().reset_index()

    # Plot
    plt.figure(figsize=(8, 6))
    sns.barplot(data=avg_best_results, x="index", y="best_value", palette="tab10")

    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel(f"Average {metric} (Best per Season)")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    plt.show()

def highlight_positions(data, threshold, labels, above=True):
    styles = pd.DataFrame('', index=data.index, columns=data.columns)
    
    for row in range(len(list(data.index))):
        for col in range(len(list(data.columns))):
            if list(data.columns)[col] in labels:
                if above:
                    if data.iloc[row,col] >= threshold:
                        styles.iloc[row, col] = 'background-color: red;'
                else:
                    if data.iloc[row,col] <= threshold:
                        styles.iloc[row, col] = 'background-color: blue;'
    return styles

def get_top_n_indices(df: pd.DataFrame, n: int, id, metric, stage):
    # Extract numeric columns (last 6 columns)
    float_df = df.iloc[:, 4:]
    
    # Flatten the dataframe and get positions
    stacked = float_df.stack()
    
    # Get top n largest values
    top_n = stacked.nlargest(n)
    
    # Convert index to row, column positions
    indices = [(id, df.iloc[row, 0], df.iloc[row, 1], col, stage, metric, value,  df.iloc[row, -1]) for (row, col), value in top_n.items()]
    
    return indices

### SERA

def compute_adjusted_boxplot_bounds(y):
    """
    Compute adjusted boxplot bounds using medcouple (MC) for skewness adjustment.
    """
    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    iqr = q3 - q1
    mc = medcouple(y)

    if mc >= 0:
        lower = q1 - 1.5 * np.exp(-4 * mc) * iqr
        upper = q3 + 1.5 * np.exp(3 * mc) * iqr
    else:
        lower = q1 - 1.5 * np.exp(-3 * mc) * iqr
        upper = q3 + 1.5 * np.exp(4 * mc) * iqr

    median = np.median(y)
    return lower, median, upper

def create_relevance_function(y):
    """
    Create a PCHIP-based relevance function φ(y) using Tukey boxplot bounds.
    """
    lower, median, upper = compute_adjusted_boxplot_bounds(y)
    
    # Extend with actual min/max to smooth edges
    # x = np.array([min(y), lower, median, upper, max(y)])
    x = np.array([lower, median, upper, max(y)])
    # if x[1] < x[0]:
    #     x[1] = (x[0]+ x[2])/2
    relevance = np.array([0, 0, 1.0, 1.0])

    # Ensure strictly increasing x by removing duplicate values
    x_unique, idx = np.unique(x, return_index=True)
    relevance_unique = relevance[idx]
    # print("Relevance points", x_unique)
    pchip = PchipInterpolator(x_unique, relevance_unique, extrapolate=True)
    return pchip

def compute_sera(y_true, y_pred, relevance_fn, step=0.01):
    """
    Compute SERA metric given true values, predictions, and a relevance function φ(y).
    """
    t_values = np.arange(0, 1 + step, step)
    ser_t = []

    for t in t_values:
        indices = [i for i, y in enumerate(y_true) if relevance_fn(y) >= t]
        if len(indices) == 0:
            ser_t.append(0.0)
            continue
        squared_errors = [(y_pred[i] - y_true[i]) ** 2 for i in indices]
        ser_t.append(np.sum(squared_errors))

    sera_score = np.trapz(ser_t, t_values)
    return sera_score, t_values, ser_t

def piecewise_linear_phi(y, bounds):
    x1, x2, x3, x4 = tf.unstack(bounds)
    iqr = (x3-x2)
    x1, x2, x3, x4, x5, x6, x7 = x1, x2, x2+iqr/4, x2+ iqr/2, x2 + iqr*0.75, x3, x4
    y1, y2, y3, y4, y5, y6, y7 = 0, 0.2, 0.4, 0.5, 0.82, 1, 1
    return tf.where(
        y <= x1, tf.zeros_like(y),
        tf.where(
            y <= x2,  y2 + (y2-y1)/(x2-x1)*(y-x2),
            tf.where(
                y <= x3, y3 + (y3-y2)/(x3-x2)*(y-x3),
                tf.where(
                    y <= x4, y4 + (y4-y3)/(x4-x3)*(y-x4),
                    tf.where(
                        y <= x5, y5 + (y5-y4)/(x5-x4)*(y-x5),
                        tf.where(
                            y <= x6, y6 + (y6-y5)/(x6-x5)*(y-x6), 
                            tf.where(
                                y <= x7, tf.ones_like(y), tf.ones_like(y) 
                            )
                        )
                    )
                )
            )
        )
    )

def piecewise_linear_phi_2(y, bounds, initial_weight=0.1):
    x1, x2, x3, x4 = tf.unstack(bounds)
    # iqr = (x3-x2)
    # x1, x2, x3, x4, x5, x6, x7 = x1, x2, x2+iqr/4, x2+ iqr/2, x2 + iqr*0.75, x3, x4
    y1, y2, y3, y4 = initial_weight, initial_weight, 1, 1
    return tf.where(
        y <= x1, y1,
        tf.where(
            y <= x2,  y2 + (y2-y1)/(x2-x1)*(y-x2),
            tf.where(
                y <= x3, y3 + (y3-y2)/(x3-x2)*(y-x3),
                tf.where(
                    y <= x4, tf.ones_like(y), tf.ones_like(y)
                    )
                )
            )
        )

class SERA(tf.keras.losses.Loss):
    def __init__(self, bounds, T=100, name="sera_loss", fn='piecewise1', initial_weight=0.1):
        super().__init__(name=name)
        self.bounds = tf.constant(bounds, dtype=tf.float32)
        self.T = T
        self.thresholds = tf.linspace(0.0, 1.0, T + 1)
        self.fn = fn
        self.initial_w = initial_weight
        # self.relevance = relevance_fn

    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        errors = tf.square(y_pred - y_true)
        if self.fn == 'piecewise1':
            relevance = piecewise_linear_phi(y_true, self.bounds)
        elif self.fn == 'piecewise2':
            relevance = piecewise_linear_phi_2(y_true, self.bounds, initial_weight=self.initial_w)
            

        total = tf.constant(0.0, dtype=tf.float32)
        for i in range(self.T + 1):
            t = self.thresholds[i]
            mask = relevance >= t
            masked_errors = tf.boolean_mask(errors, mask)
            if tf.size(masked_errors) > 0:
                weight = 0.5 if (i == 0 or i == self.T) else 1.0
                total += weight * tf.reduce_sum(masked_errors)

        return total / tf.cast(self.T, tf.float32)

### PREDICTION

class PredictionModel():
    def __init__(self, data, season, labels, regressor, name_regressor=None, frequency="bimonthly", loss_fn="mae", whole_year=False):
        self.labels = labels
        self.season = season
        self.name_regressor = name_regressor

        self.whole_year = whole_year
        
        self.is_keras_model = hasattr(regressor, "fit") and hasattr(regressor, "predict") and hasattr(regressor, "compile")
        self.data, self.features = self.form_matrix(data)
        if self.is_keras_model:
            self.early_stopping = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
            self.custom_loss = loss_fn
        if len(self.labels) > 1 and not self.is_keras_model:
            self.regressor = MultiOutputRegressor(regressor)
        else:
            self.regressor = regressor
        # self.cv_regressor = regressor (Had it to do it label by label before refactor cv and tscv)

    def form_matrix(self, data):
        #data['Date'] = pd.to_datetime(data['Date'])
        #data['Date'] = data['Date'].dt.to_period('M').astype(str)

        features = data.columns.difference(self.labels)
        self.label_scaler = StandardScaler()
        data[self.labels] = self.label_scaler.fit_transform(data[self.labels])
        if "SVR" in self.name_regressor or self.is_keras_model:
            self.scaler_X = StandardScaler().fit(data[features])
            data[features] = self.scaler_X.transform(data[features])
        
        return data, features
    
    def reshape_for_keras(self, X):
        return np.expand_dims(X, axis=1)
    
    def calculate_relevance_function(self, y, label):
        hw_index = y[[label]].to_numpy()
        hw_index = hw_index.reshape(hw_index.shape[0])

        return create_relevance_function(hw_index)
    
    def sera_error_multioutput(self, y_true, y_pred, relevance):
        sera = np.zeros(len(list(relevance.values())))
        for i, label in enumerate(list(relevance.keys())):
            sera[i], _, _ = compute_sera(y_true[[label]].to_numpy(), y_pred[:, i], relevance[label])
        return sera
    

    def train(self, len_pred):
        X = self.data[self.features]
        y = self.data[self.labels]
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = X[:-len_pred], X[-len_pred:], y[:-len_pred], y[-len_pred:]

        relevance_fs = {label: self.calculate_relevance_function(y_train, label) for label in self.labels}
        self.training_relevance_fs = relevance_fs
        if self.is_keras_model:
            X_train = self.reshape_for_keras(X_train)
            X_test = self.reshape_for_keras(X_test)
            self.regressor.fit(X_train, y_train, epochs=200, batch_size=8, verbose=1, callbacks=[self.early_stopping], validation_data=(X_test, y_test))
            y_pred_train = self.regressor.predict(X_train)
        else:
            self.regressor.fit(X_train, y_train)
            y_pred_train = self.regressor.predict(X_train)
        self.mae_training = mean_absolute_error(y_train, y_pred_train, multioutput="raw_values")
        self.mape_training = mean_absolute_percentage_error(y_train, y_pred_train, multioutput="raw_values")
        self.r2_training = r2_score(y_train, y_pred_train, multioutput="raw_values")
        self.sera_training = self.sera_error_multioutput(y_train, y_pred_train, relevance_fs)

        self.mae_average_training = mean_absolute_error(y_train, y_pred_train)
        self.mape_average_training = mean_absolute_percentage_error(y_train, y_pred_train)
        self.r2_average_training = r2_score(y_train, y_pred_train)
        self.sera_average_training = np.mean(self.sera_training)
        return y_train, y_pred_train

    def predict(self, len_pred):
        X = self.data[self.features]
        y = self.data[self.labels]
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = X[:-len_pred], X[-len_pred:], y[:-len_pred], y[-len_pred:]
        
        relevance_fs = {label: self.calculate_relevance_function(y_train, label) for label in self.labels}
        if self.is_keras_model:
            X_test = self.reshape_for_keras(X_test)
            y_pred = self.regressor.predict(X_test)
        else:
            y_pred = self.regressor.predict(X_test)
        self.mae_pred = mean_absolute_error(y_test, y_pred, multioutput="raw_values")
        self.mape_pred = mean_absolute_percentage_error(y_test, y_pred, multioutput="raw_values")
        self.r2_pred = r2_score(y_test, y_pred, multioutput="raw_values")
        self.sera_pred = self.sera_error_multioutput(y_test, y_pred, relevance_fs)

        self.mae_average_pred = mean_absolute_error(y_test, y_pred)
        self.mape_average_pred = mean_absolute_percentage_error(y_test, y_pred)
        self.r2_average_pred = r2_score(y_test, y_pred)
        self.sera_average_pred = np.mean(self.sera_pred)

        return y_test, y_pred
    
    def compile_keras_model(self, metrics=[]):
        if metrics:
            self.regressor.compile(optimizer="adam", loss=self.custom_loss, metrics=metrics)
        else:
            self.regressor.compile(optimizer="adam", loss=self.custom_loss)
        self.regressor.save_weights('model.h5')
    
    def train_predict(self, len_pred, plot=False,label_plot=None):
        if self.is_keras_model:
            self.compile_keras_model()
        ytrains, ypredtrains = self.train(len_pred)
        ytests, ypreds = self.predict(len_pred)
        self.cross_validate()
        self.timeseries_cross_validate(len_pred, plot=plot, label_plot=label_plot)
        # if plot:
        #     self.plot_predictions(len_pred, ytrains, ypredtrains, ytests, ypreds)

    def plot_predictions(self, dates, len_pred, ytrains, ypredtrains, ytests, ypreds):
        fig, axs = plt.subplots(len(self.labels), 1, figsize=(25,15))
        train_dates, test_dates = dates[:-len_pred], dates[-len_pred:]

        for i, label in enumerate(self.labels):
            # Plot training values
            axs[i].plot(
                train_dates,
                ytrains[label],
                label="Training",
                marker='o',
                color='green',
                linestyle='-',
                linewidth=1.5
            )
            axs[i].plot(
                train_dates,
                ypredtrains[:,i],
                label="Predicted Training",
                marker='x',
                color='red',
                linestyle='-',
                linewidth=1.5
            )
            axs[i].plot(
                test_dates,
                ytests[label],
                label="Test",
                marker='o',
                color='blue',
                linestyle='-',
                linewidth=1.5
            )
            axs[i].plot(
                test_dates,
                ypreds[:,i],
                label="Predicted Test",
                marker='x',
                color='red',
                linestyle='--',
                linewidth=1.5
            )
            axs[i].set_title(f"Prediction for {self.labels[i]}")
            axs[i].legend()
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle(f"Model: {self.name_regressor}")
        plt.show()
    
    def get_metric(self, metric, stage="prediction"):
        if stage == "prediction":
            if metric == "r2":
                return [self.name_regressor, self.season, metric, stage] + list(self.r2_pred) + [self.r2_average_pred]
            elif metric == "mae":
                return [self.name_regressor, self.season, metric, stage] + list(self.mae_pred)+ [self.mae_average_pred]
            elif metric == "mape":
                return [self.name_regressor, self.season, metric, stage] + list(self.mape_pred)+ [self.mape_average_pred]
            elif metric == "sera":
                return [self.name_regressor, self.season, metric, stage] + list(self.sera_pred)+ [self.sera_average_pred]

        elif stage=="training":
            if metric == "r2":
                return [self.name_regressor, self.season, metric, stage] + list(self.r2_training)+ [self.r2_average_training]
            elif metric == "mae":
                return [self.name_regressor, self.season, metric, stage] + list(self.mae_training)+ [self.mae_average_training]
            elif metric == "mape":
                return [self.name_regressor, self.season, metric, stage] + list(self.mape_training)+ [self.mape_average_training]
            elif metric == "sera":
                return [self.name_regressor, self.season, metric, stage] + list(self.sera_training)+ [self.sera_average_training]
        elif stage=="CV":
            if metric=="r2":
                return [self.name_regressor, self.season, metric, "CV"] + list(self.cv_r2_score) + [self.cv_r2_score_average]
            elif metric == "mape":
                return [self.name_regressor, self.season, metric, "CV"] + list(self.cv_mape_score) + [self.cv_mape_score_average]
            elif metric == "mae":
                return [self.name_regressor, self.season, metric, "CV"] + list(self.cv_mae_score) + [self.cv_mae_score_average]
            elif metric == "sera":
                return [self.name_regressor, self.season, metric, "CV"] + list(self.cv_sera_score)+ [self.cv_sera_score_average]
        elif stage=="TSCV":
            if metric=="r2":
                return [self.name_regressor, self.season, metric, "TSCV"] + list(self.tscv_r2_score) + [self.tscv_r2_score_average]
            elif metric == "mape":
                return [self.name_regressor, self.season, metric, "TSCV"] + list(self.tscv_mape_score) + [self.tscv_mape_score_average]
            elif metric == "mae":
                return [self.name_regressor, self.season, metric, "TSCV"] + list(self.tscv_mae_score) + [self.tscv_mae_score_average]
            elif metric == "sera":
                return [self.name_regressor, self.season, metric, "TSCV"] + list(self.tscv_sera_score)+ [self.tscv_sera_score_average]
    
    def cross_validate(self, cv=10):
        r2_cv = []
        mape_cv = []
        mae_cv = []
        sera_cv = []
        # if not self.is_keras_model:
        #     for label in self.labels:
        #         X, y = self.data[self.features], self.data[label]
        #         r2_cv.append(cross_val_score(self.cv_regressor, X, y, cv=cv, scoring='r2').mean())
        #         mape_cv.append(-1*cross_val_score(self.cv_regressor, X, y, cv=cv, scoring='neg_mean_absolute_percentage_error').mean())
        #         mae_cv.append(-1*cross_val_score(self.cv_regressor, X, y, cv=cv, scoring='neg_mean_absolute_error').mean())
        # else:
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        X, y = self.data[self.features], self.data[self.labels]
        train_sizes = []
        for train_index, test_index in kf.split(X):
            train_sizes.append(len(train_index))
            train_data, test_data, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
            relevance_fs = {label: self.calculate_relevance_function(y_train, label) for label in self.labels}
            if self.is_keras_model:
                X_train = self.reshape_for_keras(train_data)
                X_test = self.reshape_for_keras(test_data)
                self.regressor.load_weights('model.h5')
                self.early_stopping = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
                self.regressor.fit(X_train, y_train, epochs=200, batch_size=8, verbose=0, callbacks=[self.early_stopping], validation_data=(X_test, y_test))
                pred = self.regressor.predict(X_test)
            else:
                self.regressor.fit(train_data, y_train)
                pred = self.regressor.predict(test_data)
            r2_cv.append(r2_score(y_test, pred, multioutput="raw_values")) 
            mape_cv.append(mean_absolute_percentage_error(y_test, pred, multioutput="raw_values"))
            mae_cv.append(mean_absolute_error(y_test, pred, multioutput="raw_values"))
            sera_cv.append(self.sera_error_multioutput(y_test, pred, relevance_fs))
        
        r2_folds = np.array(r2_cv)
        mape_folds = np.array(mape_cv)
        mae_folds = np.array(mae_cv)
        sera_folds = np.array(sera_cv)
        weights = np.array(train_sizes)/np.sum(train_sizes)
        r2_cv = np.dot(r2_folds.T,weights)
        mape_cv = np.dot(mape_folds.T,weights)
        mae_cv = np.dot(mae_folds.T,weights)
        sera_cv = np.dot(sera_folds.T, weights)

        self.cv_r2_score = r2_cv
        self.cv_r2_score_average = np.mean(self.cv_r2_score)
        self.cv_mape_score = mape_cv
        self.cv_mape_score_average = np.mean(self.cv_mape_score)
        self.cv_mae_score = mae_cv
        self.cv_mae_score_average = np.mean(self.cv_mae_score)
        self.cv_sera_score = sera_cv
        self.cv_sera_score_average = np.mean(self.cv_sera_score)
    
    def timeseries_cross_validate(self, len_pred, plot=False, label_plot=None):
        r2_tscv = []
        mape_tscv= []
        mae_tscv= []
        sera_tscv = []
        tscv = TimeSeriesSplit(test_size=len_pred)
        if label_plot:
            cv_dates_list = []
            cv_ytrains_list = []
            cv_ypredtrains_list = []
            cv_ytests_list = []
            cv_ypreds_list = []
            idx_label = self.labels.index(label_plot)

        # if not self.is_keras_model:
        #     for label in self.labels:
        #         X, y = self.data[self.features], self.data[label]
        #         r2_tscv_label = []
        #         mape_tscv_label = []
        #         mae_tscv_label = []
        #         sera_tscv_label = []
        #         train_sizes = []
        #         for train_index, test_index in tscv.split(X):
        #             train_sizes.append(len(train_index))
        #             train_data, test_data, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        #             relevance_fs = {label : self.calculate_relevance_function(y_train, label)}
        #             self.cv_regressor.fit(train_data, y_train)
        #             pred = self.cv_regressor.predict(test_data)
                    
        #             r2_tscv_label.append(r2_score(y_test, pred))
        #             mape_tscv_label.append(mean_absolute_percentage_error(y_test, pred))
        #             mae_tscv_label.append(mean_absolute_error(y_test, pred))
        #             sera_tscv_label.append(self.sera_error_multioutput(y_test, pred, relevance_fs))

        #         weights = np.array(train_sizes)/np.sum(train_sizes)
        #         r2_tscv.append(np.sum(weights*r2_tscv_label))
        #         mape_tscv.append(np.sum(weights*mape_tscv_label))
        #         mae_tscv.append(np.sum(weights*mae_tscv_label))
        #         sera_tscv.append(np.sum(weights*sera_tscv_label))
        # else:
        X, y = self.data[self.features], self.data[self.labels]
        train_sizes = []
        for train_index, test_index in tscv.split(X):
            train_sizes.append(len(train_index))
            train_data, test_data, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
            relevance_fs = {label: self.calculate_relevance_function(y_train, label) for label in self.labels}
            if self.is_keras_model:
                X_train = self.reshape_for_keras(train_data)
                X_test = self.reshape_for_keras(test_data)
                self.regressor.load_weights('model.h5')
                self.early_stopping = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
                self.regressor.fit(X_train, y_train, epochs=200, batch_size=8, verbose=0, callbacks=[self.early_stopping], validation_data=(X_test, y_test))
                pred = self.regressor.predict(X_test)
                ytrain_pred = self.regressor.predict(X_train)
            else:
                self.regressor.fit(train_data, y_train)
                pred = self.regressor.predict(test_data)
                ytrain_pred = self.regressor.predict(train_data)
            
            len_data = len(train_index) + len(test_index)
            if self.whole_year:
                dates = pd.date_range(pd.to_datetime(f"1972-01"),periods=len_data,freq=pd.offsets.MonthBegin(1))
            else:
                dates = pd.date_range(pd.to_datetime(f"1972-{self.season}"),periods=len_data,freq=pd.offsets.YearBegin(1))
            train_dates, test_dates = dates[:len(train_index)], dates[len(train_index):]
            if plot:
                self.plot_predictions(dates, len(test_index), y_train, ytrain_pred, y_test, pred)
            if label_plot:
                cv_dates_list.append((train_dates, test_dates))
                cv_ytrains_list.append(y_train.iloc[:, idx_label].values) 
                cv_ypredtrains_list.append(ytrain_pred[:, idx_label])
                cv_ytests_list.append(y_test[self.labels[idx_label]].values)
                cv_ypreds_list.append(pred[:, idx_label])
            r2_tscv.append(r2_score(y_test, pred, multioutput="raw_values")) 
            mape_tscv.append(mean_absolute_percentage_error(y_test, pred, multioutput="raw_values"))
            mae_tscv.append(mean_absolute_error(y_test, pred, multioutput="raw_values"))
            sera_tscv.append(self.sera_error_multioutput(y_test, pred, relevance_fs))
        
        r2_folds = np.array(r2_tscv)
        mape_folds = np.array(mape_tscv)
        mae_folds = np.array(mae_tscv)
        sera_folds = np.array(sera_tscv)
        weights = np.array(train_sizes)/np.sum(train_sizes)
        r2_tscv = np.dot(r2_folds.T,weights)
        mape_tscv = np.dot(mape_folds.T,weights)
        mae_tscv = np.dot(mae_folds.T,weights)
        sera_tscv = np.dot(sera_folds.T, weights)
        if label_plot:
            r2_folds = r2_folds[:, idx_label]
            mape_folds = mape_folds[:, idx_label]
            mae_folds = mae_folds[:, idx_label]
            sera_folds = sera_folds[:, idx_label]
            self.plot_cv_folds_for_label(
                cv_dates_list,
                cv_ytrains_list,
                cv_ypredtrains_list,
                cv_ytests_list,
                cv_ypreds_list,
                label=self.labels[idx_label],
                r2_per_fold=r2_folds,
                mape_per_fold=mape_folds,
                mae_per_fold=mae_folds,
                sera_per_fold=sera_folds
            )
        self.tscv_r2_score = r2_tscv
        self.tscv_r2_score_average = np.mean(r2_tscv)
        self.tscv_mape_score = mape_tscv
        self.tscv_mape_score_average = np.mean(mape_tscv)
        self.tscv_mae_score = mae_tscv
        self.tscv_mae_score_average = np.mean(mae_tscv)
        self.tscv_sera_score = sera_tscv
        self.tscv_sera_score_average = np.mean(sera_tscv)

    
    def plot_cv_folds_for_label(self, dates_list, ytrains_list, ypredtrains_list, ytests_list, ypreds_list, label, r2_per_fold=None, mape_per_fold=None, mae_per_fold=None, sera_per_fold=None):
        n_folds = len(dates_list)
        last_train_dates, last_test_dates = dates_list[-1]
        x_min = last_train_dates.min()
        x_max = last_test_dates.max() + pd.DateOffset(years=2)
        
        for fold_idx, ((train_dates, test_dates), y_train, y_pred_train, y_test, y_pred) in enumerate(
            zip(dates_list, ytrains_list, ypredtrains_list, ytests_list, ypreds_list)
        ):
            fig, ax = plt.subplots(figsize=(25, 3))

            # Plot training data
            ax.plot(train_dates, y_train,label="Training",
                marker='o',
                color='green',
                linestyle='-',
                linewidth=1.5
            )
            ax.plot(train_dates, y_pred_train,
                label="Predicted Training",
                marker='x',
                color='red',
                linestyle='-',
                linewidth=1.5
            )
            # Plot test data
            ax.plot(test_dates, y_test,
                label="Test",
                marker='o',
                color='blue',
                linestyle='-',
                linewidth=1.5
            )
            ax.plot(test_dates, y_pred,
                label="Predicted Test",
                marker='x',
                color='red',
                linestyle='--',
                linewidth=1.5
            )
            ax.set_xlim([x_min, x_max])
            r2_text = f"R²: {r2_per_fold[fold_idx]:.3f}" if r2_per_fold is not None else ""
            mae_text = f"MAE: {mae_per_fold[fold_idx]:.3f}" if mae_per_fold is not None else ""
            mape_text = f"MAPE: {mape_per_fold[fold_idx]*100:.2f}%" if mape_per_fold is not None else ""
            sera_text = f"SERA: {sera_per_fold[fold_idx]:.3f}" if sera_per_fold is not None else ""

            # Add metrics to title
            title = f"TSCV Fold {fold_idx + 1} Predictions for {label}"
            if r2_text or mape_text or mae_text:
                title += f"\n{r2_text} | {mape_text} | {mae_text} | {sera_text}"

            ax.set_title(title)
            ax.set_xlabel("Date")
            ax.set_ylabel(label)
            ax.legend(loc='upper right')
            plt.tight_layout()
            plt.show()

    def mdi_importance(self):
        if len(self.labels) == 1 and "RF" in self.name_regressor:
            forest = self.regressor
            feature_names = self.features
            importances = forest.feature_importances_
            std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)


            forest_importances = pd.Series(importances, index=feature_names)

            fig, ax = plt.subplots(figsize=(15,8))
            forest_importances.plot.bar(yerr=std, ax=ax)
            ax.set_title(f"Feature importances using MDI for {self.name_regressor} for season {self.season}")
            ax.set_ylabel("Mean decrease in impurity")
            plt.xticks(rotation=45)
            fig.tight_layout()
        else:
            return "Not implemented yet"
    
    def permutation_importance(self, len_pred):
        if len(self.labels) == 1 and "RF" in self.name_regressor:
            X = self.data[self.features]
            y = self.data[self.labels]
            X_train, X_test, y_train, y_test = X[:-len_pred], X[-len_pred:], y[:-len_pred], y[-len_pred:]
            result = permutation_importance(
                    self.regressor, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
                )
            forest_importances = pd.Series(result.importances_mean, index=self.features)
            fig, ax = plt.subplots(figsize=(15,8))
            forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
            ax.set_title(f"Feature importances using permutation on full model {self.name_regressor} for season {self.season}")
            ax.set_ylabel("Mean accuracy decrease")
            fig.tight_layout()
            plt.xticks(rotation=45)
            plt.show()
        else:
            return "Not implemented yet"
    
    def svm_importance(self):
        if len(self.labels) == 1 and "SV" in self.name_regressor:
            pd.Series(abs(self.regressor.coef_[0]), index=self.features.to_list()).plot(kind='barh')
    
    def xgb_importance(self, type='gain'):
        fig, ax = plt.subplots(figsize=(15,8))
        plot_importance(self.regressor, ax=ax, importance_type=type)
        ax.set_title(f"Feature importances ({type}) on model {self.name_regressor} for season {self.season}")
        plt.show()

    def nn_feature_importance(self, len_pred=5, n_repeats=10, plot=True):
        """
        Calculate feature importance for Keras neural network model using permutation importance.
        
        Parameters:
        -----------
        len_pred : int, optional
            Number of samples to use for prediction. If None, uses all data.
        n_repeats : int, default=10
            Number of times to permute each feature.
        plot : bool, default=True
            Whether to plot the feature importance.
        
        Returns:
        --------
        pd.Series
            Feature importance values indexed by feature names.
        """
        if not self.is_keras_model:
            return "Not a Keras model. Use appropriate importance method for this model type."
        
        # Prepare data
        X = self.data[self.features]
        y = self.data[self.labels]
        
        if len_pred is not None:
            # Use test data if len_pred is provided
            X_test = X[-len_pred:].copy()
            y_test = y[-len_pred:].copy()
        else:
            # Use all data if len_pred is None
            X_test = X.copy()
            y_test = y.copy()
        
        # Reshape for Keras
        X_test_reshaped = self.reshape_for_keras(X_test)
        
        # Calculate baseline score
        baseline_y_pred = self.regressor.predict(X_test_reshaped)
        baseline_score = mean_absolute_error(y_test, baseline_y_pred)
        
        # Calculate importance for each feature
        importances = {}
        importances_std = {}
        
        for feature in self.features:
            feature_importance = []
            for _ in range(n_repeats):
                # Create a copy of the data and shuffle the feature
                X_permuted = X_test.copy()
                X_permuted[feature] = np.random.permutation(X_permuted[feature].values)
                
                # Reshape for Keras
                X_permuted_reshaped = self.reshape_for_keras(X_permuted)
                
                # Predict and calculate score
                permuted_y_pred = self.regressor.predict(X_permuted_reshaped)
                permuted_score = mean_absolute_error(y_test, permuted_y_pred)
                
                # The importance is the difference in score
                # Higher difference means more important feature
                feature_importance.append(permuted_score - baseline_score)
            
            importances[feature] = np.mean(feature_importance)
            importances_std[feature] = np.std(feature_importance)
        
        # Convert to pandas Series
        importance_series = pd.Series(importances)
        std_series = pd.Series(importances_std)
        
        # Sort by importance
        importance_series = importance_series.sort_values(ascending=False)
        std_series = std_series.reindex(importance_series.index)
        
        # Plot if requested
        if plot:
            fig, ax = plt.subplots(figsize=(15, 8))
            importance_series.plot.bar(yerr=std_series, ax=ax)
            ax.set_title(f"Feature importances using permutation for Keras model on season {self.season}")
            ax.set_ylabel("Mean MAE increase")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        
        return importance_series

 

class PredictionExperiment():
    """
    One PredictionExperiment correspond to one id experiment. It contains multiple results inside only bc of different metrics 
    """
    
    def __init__(self, data, labels, regressors, name_regressors, len_pred, data_id, frequency="bimonthly", loss_fn="mae", whole_year=False):
        self.seasons = data.keys()
        self.data_id = data_id
        self.labels = labels
        self.models = {season: [PredictionModel(data[season], season, labels, regressor, name_regressor=name, loss_fn=loss_fn, whole_year=whole_year) for regressor,name in zip(regressors, name_regressors)] for season in self.seasons}
        self.len_pred = len_pred

        self.num_results = 0
        self.results = pd.DataFrame(columns=["Model", "Season", "Metric", "Stage"]+ self.labels + ["Average"])
        
    def execute_experiment(self, plot=False, label_plot=None):
        for season, models in self.models.items():
            for model in models:
                print("Train predicting ", season, model.name_regressor)
                model.train_predict(self.len_pred, plot=plot, label_plot=label_plot)

    def get_metrics(self, metric, stage="prediction", thresh=0.5, above=True, show=True):
        results = pd.DataFrame(columns=["Model", "Season", "Metric", "Stage"]+ self.labels + ["Average"])
        for season in self.seasons:
            for model in self.models[season]:
                results.loc[len(results.index)] = model.get_metric(metric, stage)
        self.results = pd.concat([self.results, results])
        self.num_results += 1
        if show:
            return results.style.set_caption(f"{metric} Model Results for {stage}").format(precision=2).apply(highlight_positions, threshold=thresh, above=above, labels=self.labels, axis=None)
        else:
            return
    
    def save_results(self, path):
        df_to_save = self.results.copy()
        df_to_save["ID_data"] = self.data_id
        df_to_save.to_csv(path, mode="a", header=False, index=False)

    
    def top_results(self, metric, top_n, top_data_path=None, stage="prediction"):
        df_metric = self.results[(self.results["Metric"]==metric)&(self.results["Stage"]==stage)]
        top = pd.DataFrame(get_top_n_indices(df_metric, top_n, self.data_id, metric, stage), columns=["ID_exp", "Model", "Season", "Index", "Stage", "Metric", "Value", "Average"])
        if top_data_path:
            top.to_csv(top_data_path, mode="a", header=False, index=False)
        return top
    
    def get_feature_importance(self, season, model, method="mdi", xgbtype="gain"):
        for mod in self.models[season]:
            if mod.name_regressor == model:
                if method=="mdi":
                    mod.mdi_importance()
                elif method=="permutation":
                    mod.permutation_importance(self.len_pred)
                elif method=="svm":
                    mod.svm_importance()
                elif method=="xgboost":
                    mod.xgb_importance(xgbtype)
                elif method=="nn":
                    mod.nn_feature_importance()
                return
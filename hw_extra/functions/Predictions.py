import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.inspection import permutation_importance

from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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

def summarize_best_results_by_index(results, metadata, metric="r2", stage="prediction", top_n=1):

    # Filter for prediction stage based on metric
    prediction_results = results[(results["stage"] == stage) & (results["metric"] == metric) & (results["model"]!= "GPR-rbf-noise")]
    #prediction_results = results[(results["stage"] == stage) & (results["metric"] == metric)]


    # Find the top N best values per index (maximize for r2 and cv_r2, minimize for mape)
    best_results = prediction_results.set_index(["model", "season", "id_data"])[["HWN", "HWF", "HWD", "HWM", "HWA", "Average"]].stack().reset_index()
    best_results.columns = ["model", "season", "id_data", "index", "best_value"]
    
    if metric =="r2":
        best_results = best_results.groupby("index").apply(lambda x: x.nlargest(top_n, "best_value")).reset_index(drop=True)
    else:  # metric == "mape"
        best_results = best_results.groupby("index").apply(lambda x: x.nsmallest(top_n, "best_value")).reset_index(drop=True)

    # Get corresponding training values
    if metric in ["r2", "mape"] and stage not in ["CV","TSCV"]:
        training_results = results[(results["stage"] == "training") & (results["metric"] == metric) & results["id_data"].isin(best_results["id_data"])]
        training_results = training_results.set_index(["model", "season", "id_data"])[["HWN", "HWF", "HWD", "HWM", "HWA", "Average"]].stack().reset_index()
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

### PREDICTION

class PredictionModel():
    def __init__(self, data, season, labels, regressor, name_regressor=None, frequency="bimonthly"):
        self.labels = labels
        self.season = season
        self.name_regressor = name_regressor
        self.data, self.features = self.form_matrix(data)

        if len(self.labels) > 1:
            self.regressor = MultiOutputRegressor(regressor)
            
        else:
            self.regressor = regressor

        self.cv_regressor = regressor

    def form_matrix(self, data):
        #data['Date'] = pd.to_datetime(data['Date'])
        #data['Date'] = data['Date'].dt.to_period('M').astype(str)

        features = data.columns.difference(self.labels)
        if "SVR" in self.name_regressor:
            scaler = StandardScaler().fit(data[features])
            data[features] = scaler.transform(data[features])
        
        return data, features
    
    def train(self, len_pred):
        X = self.data[self.features]
        y = self.data[self.labels]
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = X[:-len_pred], X[-len_pred:], y[:-len_pred], y[-len_pred:]
        
        self.regressor.fit(X_train, y_train)
        y_pred_train = self.regressor.predict(X_train)
        self.mae_training = mean_absolute_error(y_train, y_pred_train, multioutput="raw_values")
        self.mape_training = mean_absolute_percentage_error(y_train, y_pred_train, multioutput="raw_values")
        self.r2_training = r2_score(y_train, y_pred_train, multioutput="raw_values")
        self.mae_average_training = mean_absolute_error(y_train, y_pred_train)
        self.mape_average_training = mean_absolute_percentage_error(y_train, y_pred_train)
        self.r2_average_training = r2_score(y_train, y_pred_train)
        
        return y_train, y_pred_train

    def predict(self, len_pred):
        X = self.data[self.features]
        y = self.data[self.labels]
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = X[:-len_pred], X[-len_pred:], y[:-len_pred], y[-len_pred:]
        
        y_pred = self.regressor.predict(X_test)
        self.mae_pred = mean_absolute_error(y_test, y_pred, multioutput="raw_values")
        self.mape_pred = mean_absolute_percentage_error(y_test, y_pred, multioutput="raw_values")
        self.r2_pred = r2_score(y_test, y_pred, multioutput="raw_values")
        self.mae_average_pred = mean_absolute_error(y_test, y_pred)
        self.mape_average_pred = mean_absolute_percentage_error(y_test, y_pred)
        self.r2_average_pred = r2_score(y_test, y_pred)

        
        return y_train, y_pred
    
    def train_predict(self, len_pred, plot=False):
        ytrains, ypredtrains = self.train(len_pred)
        ytests, ypreds = self.predict(len_pred)
        self.cross_validate()
        self.timeseries_cross_validate()
        if plot:
            self.plot_predictions(len_pred, ytrains, ypredtrains, ytests, ypreds)

    def plot_predictions(self, len_pred, ytrains, ypredtrains, ytests, ypreds):
        fig, axs = plt.subplots(len(self.labels), 1, figsize=(25,15))
        dates = self.data["Date"]
        train_dates, test_dates = dates[:-len_pred], dates[-len_pred:]

        for i in range(len(self.labels)):
            # Plot training values
            axs[i].plot(
                train_dates,
                ytrains[i],
                label="Training",
                marker='o',
                color='green',
                linestyle='-',
                linewidth=1.5
            )
            axs[i].plot(
                train_dates,
                ypredtrains[i],
                label="Predicted Training",
                marker='x',
                color='red',
                linestyle='-',
                linewidth=1.5
            )
            axs[i].plot(
                test_dates,
                ytests[i],
                label="Test",
                marker='o',
                color='blue',
                linestyle='-',
                linewidth=1.5
            )
            axs[i].plot(
                test_dates,
                ypreds[i],
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

        elif stage=="training":
            if metric == "r2":
                return [self.name_regressor, self.season, metric, stage] + list(self.r2_training)+ [self.r2_average_training]
            elif metric == "mae":
                return [self.name_regressor, self.season, metric, stage] + list(self.mae_training)+ [self.mae_average_training]
            elif metric == "mape":
                return [self.name_regressor, self.season, metric, stage] + list(self.mape_training)+ [self.mape_average_training]
        elif stage=="CV":
            if metric=="r2":
                return [self.name_regressor, self.season, metric, "CV"] + list(self.cv_r2_score) + [self.cv_r2_score_average]
            elif metric == "mape":
                return [self.name_regressor, self.season, metric, "CV"] + list(self.cv_mape_score) + [self.cv_mape_score_average]
        elif stage=="TSCV":
            if metric=="r2":
                return [self.name_regressor, self.season, metric, "TSCV"] + list(self.tscv_r2_score) + [self.tscv_r2_score_average]
            elif metric == "mape":
                return [self.name_regressor, self.season, metric, "TSCV"] + list(self.tscv_mape_score) + [self.tscv_mape_score_average]
    
    def cross_validate(self, cv=10):
        r2_cv = []
        mape_cv = []
        for label in self.labels:
            X, y = self.data[self.features], self.data[label]
            r2_cv.append(cross_val_score(self.cv_regressor, X, y, cv=cv, scoring='r2').mean())
            mape_cv.append(cross_val_score(self.cv_regressor, X, y, cv=cv, scoring='neg_mean_absolute_percentage_error').mean())
        self.cv_r2_score = r2_cv
        self.cv_r2_score_average = np.mean(self.cv_r2_score)
        self.cv_mape_score = mape_cv
        self.cv_mape_score_average = np.mean(self.cv_mape_score)
    
    def timeseries_cross_validate(self):
        r2_tscv = []
        mape_tscv= []
        tscv = TimeSeriesSplit(test_size=5)
        for label in self.labels:
            X, y = self.data[self.features], self.data[label]
            r2_tscv_label = []
            mape_tscv_label = []
            train_sizes = []
            for train_index, test_index in tscv.split(X):
                train_sizes.append(len(train_index))
                train_data, test_data, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
                self.cv_regressor.fit(train_data, y_train)
                pred = self.cv_regressor.predict(test_data)
                r2_tscv_label.append(r2_score(y_test, pred))
                mape_tscv_label.append(mean_absolute_percentage_error(y_test, pred))
            weights = np.array(train_sizes)/np.sum(train_sizes)
            r2_tscv.append(np.sum(weights*r2_tscv_label))
            mape_tscv.append(np.sum(weights*mape_tscv_label))
        self.tscv_r2_score = r2_tscv
        self.tscv_r2_score_average = np.mean(r2_tscv)
        self.tscv_mape_score = mape_tscv
        self.tscv_mape_score_average = np.mean(mape_tscv)

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


 

class PredictionExperiment():
    """
    One PredictionExperiment correspond to one id experiment. It contains multiple results inside only bc of different metrics 
    """
    
    def __init__(self, data, labels, regressors, name_regressors, len_pred, data_id, frequency="bimonthly"):
        self.seasons = data.keys()
        self.data_id = data_id
        self.labels = labels
        self.models = {season: [PredictionModel(data[season], season, labels, regressor, name_regressor=name) for regressor,name in zip(regressors, name_regressors)] for season in self.seasons}
        self.len_pred = len_pred

        self.num_results = 0
        self.results = pd.DataFrame(columns=["Model", "Season", "Metric", "Stage"]+ self.labels + ["Average"])
        
    
    def execute_experiment(self):
        for season, models in self.models.items():
            for model in models:
                model.train_predict(self.len_pred)

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
    
    def get_feature_importance(self, season, model, method="mdi"):
        for mod in self.models[season]:
            if mod.name_regressor == model:
                if method=="mdi":
                    mod.mdi_importance()
                elif method=="permutation":
                    mod.permutation_importance(self.len_pred)
                elif method=="svm":
                    mod.svm_importance()
                return
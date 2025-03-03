import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.multioutput import MultiOutputRegressor


import matplotlib.pyplot as plt

### UTILS

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

def get_top_n_indices(df: pd.DataFrame, n: int = 3):
    # Extract numeric columns (last 6 columns)
    float_df = df.iloc[:, 4:]
    
    # Flatten the dataframe and get positions
    stacked = float_df.stack()
    
    # Get top n largest values
    top_n = stacked.nlargest(n)
    
    # Convert index to row, column positions
    indices = [(df.iloc[row, 0], df.iloc[row, 1], col, value) for (row, col), value in top_n.items()]
    
    return indices

### PREDICTION

class PredictionModel():
    def __init__(self, data, season, labels, regressor, name_regressor=None, frequency="bimonthly"):
        self.labels = labels
        self.season = season

        self.data, self.features = self.form_matrix(data)
        
        if len(self.labels) > 1:
            self.regressor = MultiOutputRegressor(regressor)
        else:
            self.regressor = regressor
        self.name_regressor = name_regressor

    def form_matrix(self, data):
        #data['Date'] = pd.to_datetime(data['Date'])
        #data['Date'] = data['Date'].dt.to_period('M').astype(str)

        features = data.columns.difference(self.labels)
        
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
        if metric == "cv_r2":
            return [self.name_regressor, self.season, metric, "CV"] + list(self.cv_score) + [self.cv_score_average]
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
    
    def cross_validate(self, cv=5):
        X, y = self.data[self.features], self.data[self.labels]
        self.cv_score = cross_val_score(self.regressor, X, y, cv=cv, scoring='r2')
        self.cv_score_average = self.cv_score.mean()


class PredictionExperiment():
    """
    One PredictionExperiment correspond to one id experiment. It contains multiple results inside only bc of different metrics 
    """
    
    def __init__(self, data, labels, regressors, name_regressors, len_pred, frequency="bimonthly"):
        self.seasons = data.keys()
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
    
    def top_results(self, metric, top_n, stage="prediction"):
        df_metric = self.results[self.results["Metric"]==metric]
        top = pd.DataFrame(get_top_n_indices(df_metric, top_n), columns=["Model", "Season", "Index", "Value"])
        return top
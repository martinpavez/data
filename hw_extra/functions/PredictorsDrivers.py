import pandas as pd
import numpy as np
from itertools import combinations

import re
import pickle

from IndexDrivers import MultivariatePCA

### AUXILIAR FUNCTIONS

def compute_correlations(df, num_indices=5, method='pearson'):
    """
    Compute the correlation between the first `num_indices` columns and the rest of the columns in a DataFrame
    using the specified method (Pearson, Spearman, or Kendall).
    
    Parameters:
    - df: pd.DataFrame, the input DataFrame with timeseries as columns.
    - num_indices: int, number of columns to use as the principal indices for correlation.
    - method: str, the method for correlation ('pearson', 'spearman', 'kendall').
    
    Returns:
    - pd.DataFrame with correlations between the `num_indices` columns and the rest of the columns.
    """
    if method not in ['pearson', 'spearman', 'kendall']:
        raise ValueError("Method must be 'pearson', 'spearman', or 'kendall'")

    # Extract the first `num_indices` columns as principal indices
    principal_df = df.iloc[:, :num_indices]
    
    # Extract the rest of the columns
    other_df = df.iloc[:, num_indices:]
    
    # If method is not 'pearson', we need to use rank correlation methods
    if method == 'spearman':
        # Rank the data for Spearman correlation
        principal_df = principal_df.rank()
        other_df = other_df.rank()
    elif method == 'kendall':
        # Pandas `.corr()` will be used for Kendall to handle pairwise calculation
        return other_df.corrwith(principal_df, axis=0, method='kendall')
    
    # Compute correlation matrix using dot product for Pearson or Spearman
    principal_values = principal_df.values
    other_values = other_df.values
    
    # Mean-center the data
    principal_mean_centered = principal_values - principal_values.mean(axis=0)
    other_mean_centered = other_values - other_values.mean(axis=0)
    
    # Compute the standard deviations
    principal_std = principal_values.std(axis=0)
    other_std = other_values.std(axis=0)
    
    # Normalize data
    principal_normalized = principal_mean_centered / principal_std
    other_normalized = other_mean_centered / other_std
    
    # Compute the correlation matrix using dot product
    correlations = np.dot(other_normalized.T, principal_normalized) / (other_values.shape[0] - 1)
    
    # Convert to DataFrame for readability, using original column names
    correlation_df = pd.DataFrame(
        correlations,
        columns=df.columns[:num_indices],  # Names of the principal indices columns
        index=df.columns[num_indices:]     # Names of the remaining columns
    )
    
    return correlation_df

def build_pos_neg_corr_df(data, label_columns, methods=["pearson","spearman"]):
    correlations = { met: compute_correlations(data, num_indices=len(label_columns),method=met) for met in methods}
    any_correlations = {
        "positive": {met: [] for met in methods},
        "negative": {met: [] for met in methods}
    }
    # Iterate over each correlation matrix and collect pairs
    for method, corr_df in correlations.items():
        for timeserie, row in corr_df.iterrows():
            for index, value in row.items():
                if value > 0:
                    any_correlations["positive"][method].append((timeserie, index, value))
                elif value < 0:
                    any_correlations["negative"][method].append((timeserie, index, value))

    positive_corr_any_df = {
        method: pd.DataFrame(pairs, columns=["Timeserie", "Index", "Correlation"])
        for method, pairs in any_correlations["positive"].items()
    }
    negative_corr_any_df = {
        method: pd.DataFrame(pairs, columns=["Timeserie", "Index", "Correlation"])
        for method, pairs in any_correlations["negative"].items()
    }
    return [positive_corr_any_df, negative_corr_any_df]

def build_df_correlations_monthly(dict_predictors, season_corr_dict, methods=["pearson", "spearman"]):
    any_correlations_list = []
    for month, (positive_corr_dict, negative_corr_dict) in season_corr_dict.items():
        # Combine positive and negative correlations for each method
        for method in methods:
            if method in positive_corr_dict:
                for _, row in positive_corr_dict[method].iterrows():
                    any_correlations_list.append((month, method, *row))
            
            if method in negative_corr_dict:
                for _, row in negative_corr_dict[method].iterrows():
                    any_correlations_list.append((month, method, *row))
    # Convert the collected data to a DataFrame for easier processing
    correlations_df = pd.DataFrame(
        any_correlations_list, columns=["Season", "Method", "PC", "Index", "Correlation"]
    )
    correlations_df['ID'] = correlations_df['PC'].apply(lambda x: re.search(r'PC_(.*?)-Mode-', x).group(1))
    correlations_df['Variance'] = correlations_df.apply(lambda x: dict_predictors[int(x.ID)].explained_variance[str(x.Season)][int(x.PC[-1])-1], axis=1)

    return correlations_df

def build_df_correlations_yearly(dict_predictors, pos_neg_df, methods=["pearson", "spearman"]):
    all_correlations = []

    for method in ["pearson", "spearman"]:
        if method in pos_neg_df[0]:
            for _, row in pos_neg_df[0][method].iterrows():
                all_correlations.append((method, *row))
        
        if method in pos_neg_df[1]:
            for _, row in pos_neg_df[1][method].iterrows():
                all_correlations.append(( method, *row))

    # Convert the collected data to a DataFrame for easier processing
    correlations_df = pd.DataFrame(
        all_correlations, columns=["Method", "PC", "Index", "Correlation"]
    )
    correlations_df['ID'] = correlations_df['PC'].apply(lambda x: re.search(r'PC_(.*?)-Mode-', x).group(1))
    # Sort by the absolute value of the correlation and get the top 10

    ## Include explained variance
    correlations_df['Variance'] = correlations_df.apply(lambda x: dict_predictors[int(x.ID)].explained_variance[int(x.PC[-1])-1], axis=1)
    return correlations_df

def get_top_corr(all_corr_df, top_n):
    top_correlations = all_corr_df.reindex(
        all_corr_df["Correlation"].abs().sort_values(ascending=False).index
    )
    top_list = []
    for pc in list(top_correlations["PC"]):
        if len(top_list)==top_n:
            break
        elif pc not in top_list:
            top_list.append(pc)
    return top_list

### PREDICTORS FUNCTIONS

class Predictor():
    def __init__(self):
        pass

    def calculate_predictors(self):
        pass

    def get_predictors(self):
        pass


class PCAPredictors(Predictor):
    def __init__(self, data, num_modes=3, boxes=None,
                total_variables=["SST", "SP", "TTR", "U10", "V10", "Z"],
                rolling_window = 2, frequency="monthly",
                start_year=1972, end_year=2022, saved_pcas = None):
        self.data = data
        self.num_modes = num_modes
        self.rolling_window = rolling_window
        self.frequency = frequency
        self.start_year = start_year
        self.end_year = end_year
        if boxes:
            self.boxes = boxes
        else:
            self.boxes = self.default_boxes()
        
        self.variables = total_variables
        if saved_pcas:
            self.dict_predictors = saved_pcas
        else:
            self.dict_predictors = self.calculate_predictors()
        self.df_predictors = self.set_df_predictors()
        self.df_label_predictors = None
        self.experiments = {}
        self.num_experiments = 0

    def calculate_predictors(self):
        var_combi = []

        for r in range(1, len(self.variables) + 1):
            var_combi.extend(combinations(self.variables, r))

        k = 0
        pcas = {}
        for box_id, box_coords in self.boxes.items():
            for var_combination in var_combi:
                pcas[k] = MultivariatePCA(self.data, self.num_modes, [self.start_year, self.end_year], box_limit=box_coords, 
                                          variables=var_combination, rolling_window=self.rolling_window,
                                          frequency=self.frequency)

                k +=1
        return pcas



    def get_predictors(self, start_year = None, end_year=None):
        df = self.df_predictors
        if start_year:
            if (start_year-self.start_year < 0):
                print("Can not retrieve for not calculated years")
                return
            else:
                df = df[df.index.year >= start_year]
        if end_year:
            if (self.end_year-end_year < 0):
                print("Can not retrieve for not calculated years")
                return
            else:
                df = df[df.index.year <= end_year]
        return df
    

    def top_correlations_predictors(self, label_data, top_n=10,
                                    methods = ["pearson", "spearman"], threshold_corr=None, 
                                    threshold_variance = None, save=False, save_path=None):
        
        self.experiments[self.num_experiments] = {}
        self.df_label_predictors = pd.concat([label_data, self.df_predictors], axis=1)
        if self.frequency=="monthly":
            seasons_any_corr = {}

            for i in range(12):
                hwis_month = self.df_label_predictors[self.df_label_predictors.index.month == i+1]
                
                seasons_any_corr[i+1] = build_pos_neg_corr_df(hwis_month, list(label_data.columns), methods=methods)

            any_correlations_df = build_df_correlations_monthly(self.dict_predictors, seasons_any_corr, methods=methods)

            if threshold_corr:
                corr_positive = any_correlations_df[any_correlations_df["Correlation"] >= threshold_corr]
                corr_negative = any_correlations_df[any_correlations_df["Correlation"] <= -1*threshold_corr]
                any_correlations_df = pd.concat([corr_positive, corr_negative], axis=0)
            if threshold_variance:
                any_correlations_df = any_correlations_df[any_correlations_df["Variance"] >= threshold_variance]

            top_corr_per_season= { }
            for i in range(1,13):
                month_corr = any_correlations_df[any_correlations_df["Season"]==i]
                top_corr_per_season[i] = get_top_corr(month_corr, top_n)
                df_top_predictors = self.df_label_predictors[self.df_label_predictors.index.month==i][list(label_data.columns) + top_corr_per_season[i]]
                self.experiments[self.num_experiments][i] = df_top_predictors

        elif self.frequency=="yearly":
            pos_df, neg_df = build_pos_neg_corr_df(self.df_label_predictors, list(label_data.columns), methods=methods)
            all_correlations_df = build_df_correlations_yearly(self.dict_predictors, [pos_df, neg_df], methods=methods)
            if threshold_corr:
                corr_positive = all_correlations_df[all_correlations_df["Correlation"] >= threshold_corr]
                corr_negative = all_correlations_df[all_correlations_df["Correlation"] <= -1*threshold_corr]
                all_correlations_df = pd.concat([corr_positive, corr_negative], axis=0)
            if threshold_variance:
                all_correlations_df = all_correlations_df[all_correlations_df["Variance"] >= threshold_variance]
            
            top_corr = get_top_corr(all_correlations_df, top_n)

            df_top_predictors = self.df_label_predictors[list(label_data.columns) + top_corr]
            self.experiments[self.num_experiments] = df_top_predictors
        
        self.num_experiments += 1
        return self.experiments[self.num_experiments-1], self.num_experiments-1
    
    def experiment_to_csv(self, experiment, folder_path, name):
        dfs = self.experiments[experiment]
        if self.frequency == "monthly":
            for season, df in dfs.items():
                df.to_csv(f"{folder_path}/{name}_{season}.csv")
        elif self.frequency=="yearly":
            dfs.to_csv(f"{folder_path}/{name}.csv")
        print("Saved")
        return
    
    def set_df_predictors(self):
        if self.frequency=="monthly":
            df = pd.DataFrame(index=pd.date_range(pd.to_datetime(f"{self.start_year}-01-01"),pd.to_datetime(f"{self.end_year}-12-01"),freq=pd.offsets.MonthBegin(1)))
        elif self.frequency=="yearly":
            df = pd.DataFrame(index=pd.date_range(pd.to_datetime(f"{self.start_year}-01-01"),pd.to_datetime(f"{self.end_year}-01-01"),freq=pd.offsets.YearBegin(1)))
        for mode in range(self.num_modes):
            for num, pca in self.dict_predictors.items():
                df[f"PC_{num}-Mode-{mode+1}"] = pca.get_index(mode+1, start_year=self.start_year, end_year=self.end_year)
        df.index.name = "Date"
        return df
    
    def incorporate_predictor(self, predictor, name, format="NOAA"):
        self.df_predictors[name] = predictor

    def default_boxes(self):
        boxes0 = {
            0: [120,205,30,60],
            1: [205,290,30,60],
            2: [120,205,0,30],
            3: [205,290,0,30],
            4: [120,205,-30,0],
            5: [205,290,-30,0],
            6: [120,205,-60,-30],
            7: [205,290,-60,-30]
            }
        # Counter for keys in boxes2
        new_key = 8
        boxes= {}

        # Iteratively split each box into two
        for key, (lon_min, lon_max, lat_min, lat_max) in boxes0.items():
            # Calculate the longitude midpoint
            lon_mid = (lon_min + lon_max) / 2
            
            # Define two new sub-boxes by splitting along the longitude midpoint
            boxes[new_key] = [lon_min, lon_mid, lat_min, lat_max]
            new_key += 1
            boxes[new_key] = [lon_mid, lon_max, lat_min, lat_max]
            new_key += 1

        boxes.update(boxes0)

        return boxes
    
    def save_predictors(self, path):
        with open(path, "wb") as inp:
            pickle.dump(self, inp, protocol=pickle.HIGHEST_PROTOCOL)
    

    

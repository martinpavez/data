import pandas as pd
import numpy as np
from itertools import combinations
import uuid
import re
import pickle

from IndexDrivers import MultivariatePCA

### AUXILIAR FUNCTIONS
def add_adjacent_month_values(df):
    """
    For each column in a date-indexed DataFrame, add two new columns containing
    the values from the previous month and the next month.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame with a DatetimeIndex at the monthly frequency
        
    Returns:
    --------
    pandas.DataFrame
        The original DataFrame with added columns for previous and next month values
    
    Example:
    --------
    If df has columns ['A', 'B'], the result will have columns 
    ['A', 'A_prev_month', 'A_next_month', 'B', 'B_prev_month', 'B_next_month']
    """
    # Make a copy to avoid modifying the original DataFrame
    result = df.copy()

    # For each column in the original DataFrame
    original_columns = list(df.columns)
    for col in original_columns:
        # Add previous month values (shift forward)
        result[f"{col}_prev_month"] = result[col].shift(1)
        
        # Add next month values (shift backward)
        result[f"{col}_next_month"] = result[col].shift(-1)
    
    # Reorder columns so each original column is followed by its prev and next month columns
    new_columns = []
    for col in original_columns:
        new_columns.extend([col, f"{col}_prev_month", f"{col}_next_month"])
    
    return result[new_columns]

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
    def __init__(self, start_year=1972, end_year=2022, frequency="monthly", stations=None):
        self.experiments = {}
        self.data_experiments = {}
        self.num_experiments = 0
        self.frequency = frequency
        self.start_year = start_year
        self.end_year = end_year
        self.stations = stations

    def calculate_predictors(self):
        pass

    def get_predictors(self):
        pass

    def create_empty_experiment(self):
        if self.frequency=="monthly":
            df = pd.DataFrame(index=pd.date_range(pd.to_datetime(f"{self.start_year}-01"),pd.to_datetime(f"{self.end_year}-12"),freq=pd.offsets.MonthBegin(1)))
            df.index.name = "Date"
            dfs = {i: df[df.index.month==i] for i in range(1,13)}
        elif self.frequency=="yearly":
            dfs = pd.DataFrame(index=pd.date_range(pd.to_datetime(f"{self.start_year}"),pd.to_datetime(f"{self.end_year}"),freq=pd.offsets.YearBegin(1)))
            dfs.index.name = "Date"
        elif self.frequency=="summerly":
            dfs = {1:pd.DataFrame(index=pd.date_range(pd.to_datetime(f"{self.start_year}"),pd.to_datetime(f"{self.end_year}"),freq=pd.offsets.YearBegin(1))) }
            dfs[1].index.name = "Date"
        self.experiments[self.num_experiments] = dfs
        self.data_experiments[self.num_experiments] = list()
        self.num_experiments += 1
        return self.experiments[self.num_experiments-1], self.num_experiments-1

    def incorporate_predictor(self, predictors, names, num_exp=0):
        '''
        Takes number of experiment and incorporates the new predictor into a new experiment with data + new predictors
        '''
        try:
            experiment = self.experiments[num_exp]
            data = self.data_experiments[num_exp]
        except KeyError:
            experiment, num = self.create_empty_experiment()
            data = self.data_experiments[num]
        if self.frequency=="monthly":
            new_exp = {}
            for season, df in experiment.items():
                df_new = df.copy()
                for name, predictor in zip(names, predictors):
                    for col in list(predictor.columns):
                        df_new[f"{name}-{col}"] = predictor[col]
                new_exp[season] = df_new
        elif self.frequency=="summerly":
            df_new = experiment[1].copy()
            predictors_summer = [add_adjacent_month_values(predictor) for predictor in predictors] 
            for name, predictor in zip(names, predictors_summer):
                for col in list(predictor.columns):
                    df_new[f"{name}-{col}"] = predictor[col]
            new_exp = {1: df_new}
        self.data_experiments[self.num_experiments] = data + ["-".join(names)]
        self.experiments[self.num_experiments] = new_exp

        self.num_experiments += 1
        return self.experiments[self.num_experiments-1], self.num_experiments-1 
    
    def incorporate_label(self, df_labels, num_exp=0):
        '''
        Takes number of experiment and incorporates the new predictor into a new experiment with data + new predictors
        '''
        try:
            experiment = self.experiments[num_exp]
            data = self.data_experiments[num_exp]
        except KeyError:
            experiment, num = self.create_empty_experiment()
            data = self.data_experiments[num]
        if self.frequency in ["monthly", "summerly"]:
            new_exp = {}
            for season, df in experiment.items():
                df_new = df.copy()
                for col in list(df_labels.columns):
                    df_new[col] = df_labels[col]
                new_exp[season] = df_new
        self.data_experiments[self.num_experiments] = data 
        self.experiments[self.num_experiments] = new_exp

        self.num_experiments += 1
        return self.experiments[self.num_experiments-1], self.num_experiments-1 

    def experiment_to_parquet(self, experiment, folder_path, metadata_path):
        dfs = self.experiments[experiment]
        data = self.data_experiments[experiment]
        if self.stations:
            data.append("-".join(self.stations))
        data_string = ",".join(data)
        id = str(uuid.uuid4())[:8]
        if self.frequency in ["monthly", "summerly"]:
            for season, df in dfs.items():
                name = f"predictor_{id}_{season}.parquet"
                df.to_parquet(f"{folder_path}/{name}")
                with open(metadata_path, "a") as file:
                    file.write(f"{id},{name},{season},{data_string}\n")
        elif self.frequency=="yearly":
            name = f"predictor_{id}.parquet"
            dfs.to_parquet(f"{folder_path}/{name}")
            with open(metadata_path, "a") as file:
                file.write(f"{id},{name},0,{data_string}\n")
        print("Saved")
        return
    



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
            self.boxes_id = 0
        
        self.variables = total_variables
        if saved_pcas:
            self.dict_predictors = saved_pcas
        else:
            self.dict_predictors = self.calculate_predictors()
        self.df_predictors = self.set_df_predictors()
        self.df_label_predictors = None
        self.experiments = {}
        self.num_experiments = 0
        self.data_experiments= {}

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
                                    threshold_variance = None):
        
        
        self.df_label_predictors = pd.concat([label_data, self.df_predictors], axis=1)
        if methods==["pearson", "spearman"]:
            label_methods = "ps"
        elif methods==["pearson"]:
            label_methods = "p"
        else:
            label_methods = "s"
        
        extra_indices = None
        if "SST" in self.variables:
            only_sea = True
        else:
            only_sea = False

        if self.frequency=="monthly":
            self.experiments[self.num_experiments] = {}
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
            self.data_experiments[self.num_experiments] = [self.boxes_id, top_n, threshold_variance, self.num_modes, f"{self.rolling_window}{self.frequency}", label_methods, extra_indices, only_sea]
                

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
            self.data_experiments[self.num_experiments] = [self.boxes_id, top_n, threshold_variance, self.num_modes, self.frequency, label_methods, extra_indices, only_sea]
        
        ## save data experiment for metadata
        
        self.num_experiments += 1
        return self.experiments[self.num_experiments-1], self.num_experiments-1
    
    def experiment_to_parquet(self, experiment, folder_path, metadata_path):
        dfs = self.experiments[experiment]
        boxes, top_n, var_thresh, modes, freq, met, extra, sea = self.data_experiments[experiment]
        id = str(uuid.uuid4())[:8]
        if self.frequency == "monthly":
            for season, df in dfs.items():
                name = f"predictor_{id}_{season}.parquet"
                df.to_parquet(f"{folder_path}/{name}")
                with open(metadata_path, "a") as file:
                    file.write(f"{id},{name},{boxes},{top_n},{var_thresh},{modes},{freq},{season},{met},{extra},{sea}\n")
        elif self.frequency=="yearly":
            name = f"predictor_{id}.parquet"
            dfs.to_parquet(f"{folder_path}/{name}")
            with open(metadata_path, "a") as file:
                file.write(f"{id},{name},{boxes},{top_n},{var_thresh},{modes},{freq},0,{met},{extra},{sea}\n")
        print("Saved")
        return
    
    def set_df_predictors(self):
        if self.frequency=="monthly":
            df = pd.DataFrame(index=pd.date_range(pd.to_datetime(f"{self.start_year}-01"),pd.to_datetime(f"{self.end_year}-12"),freq=pd.offsets.MonthBegin(1)))
        elif self.frequency=="yearly":
            df = pd.DataFrame(index=pd.date_range(pd.to_datetime(f"{self.start_year}"),pd.to_datetime(f"{self.end_year}"),freq=pd.offsets.YearBegin(1)))
        for mode in range(self.num_modes):
            for num, pca in self.dict_predictors.items():
                df[f"PC_{num}-Mode-{mode+1}"] = pca.get_index(mode+1, start_year=self.start_year, end_year=self.end_year)
        df.index.name = "Date"
        return df
    
    def parquet_to_experiment(self, uid, folder_path, metadata):
        exp = metadata[metadata["id"]==uid]
        if self.frequency=="monthly":
            self.experiments[self.num_experiments] = {}
            flatten_values = list(exp.iloc[0].values.flatten())
            self.data_experiments[self.num_experiments] = [flatten_values[2], flatten_values[3], flatten_values[4], flatten_values[5], flatten_values[6], flatten_values[8], flatten_values[9], flatten_values[10]]
            for index, row in exp.iterrows():
                season = row["season"]
                self.experiments[self.num_experiments][int(season)] = pd.read_parquet(f"{folder_path}/predictor_{uid}_{season}.parquet")
            self.num_experiments += 1
            return self.experiments[self.num_experiments-1], self.num_experiments-1
        elif self.frequency=="yearly":
            flatten_values = list(exp.values.flatten())
            #construct self.experiment and self.dataexp with self.numexp
            self.data_experiments[self.num_experiments] = [flatten_values[2], flatten_values[3], flatten_values[4], flatten_values[5], flatten_values[6], flatten_values[8], flatten_values[9], flatten_values[10]]
            self.experiments[self.num_experiments] = pd.read_parquet(f"{folder_path}/predictor_{uid}.parquet")

            self.num_experiments += 1
            return self.experiments[self.num_experiments-1], self.num_experiments-1
        
    def load_experiments(self, folder_path, metadata_path):
        metadata = pd.read_csv(metadata_path)
        if self.frequency=="monthly":
            unique_ids = metadata[metadata["frequency"]==f"{self.rolling_window}{self.frequency}"]["id"].unique()
        else:
            unique_ids = metadata[metadata["frequency"]==f"{self.frequency}"]["id"].unique()
        for id in unique_ids:
            self.parquet_to_experiment(id, folder_path, metadata)
        print("Loaded",len(list(self.experiments.keys())), "experiments")
    
    
    def incorporate_predictor(self, num_exp, predictors, names):
        '''
        Takes number of experiment and incorporates the new predictor into a new experiment with data + new predictors
        '''
        experiment = self.experiments[num_exp]
        boxes, top_n, threshold_variance, num_modes, freq, label_methods, extra, sea = self.data_experiments[num_exp]
        if self.frequency=="monthly":
            new_exp = {}
            for season, df in experiment.items():
                df_new = df.copy()
                for name, predictor in zip(names, predictors):
                    df_new[name] = predictor
                new_exp[season] = df_new
        self.data_experiments[self.num_experiments] = [boxes, top_n, threshold_variance, num_modes, freq, label_methods, "-".join(names), sea]
        self.experiments[self.num_experiments] = new_exp

        self.num_experiments += 1
        return self.experiments[self.num_experiments-1], self.num_experiments-1 # CHECK IF SAVE IS CORRECT AFTER INCORPORATING (save from outside)


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
    

    

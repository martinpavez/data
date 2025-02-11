import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from sklearn.preprocessing import MinMaxScaler
import scipy as sp
import matplotlib.cm as cm  # For colormap
import matplotlib.colors as mcolors  # For normalization

# Doing design according to functionalities applied to a dataset,
# but another design could be heritance from a xr.Dataset class and 
# define additional functions to it.


## Auxiliar functions
def drop_norm(vec: np.ndarray):
    """
    Removes columns with NaN values and applies Min-Max normalization.

    Parameters:
    vec : 2D array where normalization is applied after removing invalid columns.

    Returns:
    tuple:
        - Transformed (normalized) 2D numpy array.
        - Boolean mask of valid columns. Used to reconstruct grid after.
    """
    valid_columns = ~np.isnan(vec).any(axis=0)  # Identify valid columns (non-NaN)
    vec = vec[:, valid_columns]  #DROP

    scaler = MinMaxScaler()
    scaler.fit(vec)
    return scaler.transform(vec), valid_columns

def is_month(month, n):
    return month == n
    
def flatten_vector(ds: xr.Dataset, length: int):
    """
    Flattens multiple variables from the dataset into a single vector, removing NaNs, 
    and detrends the data.

    Parameters:
    ds : Dataset containing variables
    length : Number of time steps in the dataset.

    Returns:
    tuple:
        - Flattened and detrended numpy array with all variables combined.
        - Boolean mask indicating valid columns.
    """

    flatten, valid = drop_norm(ds[f"{list(ds.keys())[0]}"].to_numpy().reshape(length, -1))
    if len(list(ds.keys())) > 1:
        for var in list(ds.keys())[1:]:
            aux_flatten, aux_valid = drop_norm(ds[f"{var}"].to_numpy().reshape(length, -1))
            flatten = np.concatenate((flatten, aux_flatten), axis=1)
            valid = np.concatenate((valid, aux_valid))
    
    flatten = detrend_data(flatten)
    flatten = flatten -np.mean(flatten, axis=0)

    return flatten, valid


def detrend_data(matrix):
    """
    Removes linear trends from each variable (column) in the input matrix.
    """
    n_timesteps, n_variables = matrix.shape
    time = np.arange(n_timesteps)
    detrended_matrix = np.zeros_like(matrix)
    
    # Detrend each column (each time series) individually
    for i in range(n_variables):
        y = matrix[:, i]
        
        # Perform a linear regression to get the slope and intercept
        slope, intercept, _, _, _ = sp.stats.linregress(time, y)
        trend = slope * time + intercept
        
        # Subtract the trend from the original data
        detrended_matrix[:, i] = y - trend
    
    return detrended_matrix

## Index Definitions

class Index():
    """
    Base class for different types of climate indices.
    """
    def __init__(self, data: xr.Dataset):
        ##Time series value of index
        self.index = None 
        self.data = data
    
    def calculate_index(self):
        pass
    
    def get_index(self):
        return self.index
    
class MultivariatePCA(Index):
    """
    Computes multivariate Principal Component Analysis (PCA) for multiple variables.

    Parameters:
    data : Input dataset containing climate variables.
    n_modes :  Number of principal components (modes) to retain.
    reference_period : List containing start and end year for the reference period.
    variables : List of variables included in the analysis.
    box_limit :  Geographic bounds for longitude and latitude

    Inherits:
    Index : Base class for climate indices.
    """

    def __init__(self, data: xr.Dataset, n_modes: int, reference_period:list,
                  variables=["SST", "SP", "TTR", "U10", "V10"], box_limit=[100,290,-30,30],
                    rolling_window=2, frequency="monthly"):
        super().__init__(data)
        self.data = self.data.sel(longitude=slice(box_limit[0], box_limit[1]), latitude=slice(box_limit[2],box_limit[3]))
        self.variables_dict = { i+1: variables[i] for i in range(len(variables))}

        self.filter_variables() #filter according to variables

        self.n_modes = n_modes
        self.start_year = reference_period[0]
        self.end_year = reference_period[1]
        self.pcs = {} #dict: season -> array of k-pcs 
        self.modes = {} #dict: season -> array of k-modes 
        self.explained_variance = {}
        self.grid_shape = (self.data.sizes["latitude"], self.data.sizes["longitude"])
        self.n_variables = len(variables)
        self.rolling_window = rolling_window
        self.freq = frequency

        self.box = box_limit
        if frequency=="monthly":
            self.calculate_index_monthly()
        elif frequency=="yearly":
            self.calculate_index_yearly()
    
    def __str__(self):
        return f"PCA on box {self.box} with variables {list(self.variables_dict.values())}"
        

    def calculate_index_monthly(self):
        """
        Calculates PCA index for each month and stores principal components and explained variance.
        """
        for i in range(1,13):
            self.pcs[f"{i}"], self.explained_variance[f"{i}"], modes, columns = self.calculate_index_by_season(i)
            self.modes[f"{i}"] = self.map_modes_to_full_grid(modes, columns)

        self.correct_pca()
        
        self.get_full_index()

    def calculate_index_yearly(self):
        yearly = self.data.groupby('time.year').mean('time')
        flatten_yearly, valid_columns = flatten_vector(yearly, yearly.sizes["year"])

        cov_matrix = np.matmul(flatten_yearly, flatten_yearly.T)/(flatten_yearly.shape[0]-1)
        N = cov_matrix.shape[0]
        D, V= sp.linalg.eigh(cov_matrix, subset_by_index=[N-self.n_modes,N-1])

        # Reverse for descending order
        D = D[::-1]
        V = V[:, ::-1]

        pcs = []
        modes = []

        for i in range(self.n_modes):
            # Project kth eigenvector onto the original space
            V_k = np.matmul(flatten_yearly.T, V[:, i])

            # Normalize the mode
            sq = np.sqrt(D[i])
            V_k = V_k / sq

            # Compute the principal component time series
            pc_k = V_k.T @ flatten_yearly.T / (np.dot(V_k.T, V_k))

            pcs.append(pc_k)
            modes.append(V_k)
        
        self.pcs = np.array(pcs)
        self.explained_variance = D / np.trace(cov_matrix)
        self.modes = self.map_modes_to_full_grid(np.array(modes), valid_columns)

        self.get_full_index_yearly()
    
    def correct_pca(self):
        """
        Corrects the signs of consecutive principal components to maintain consistency across months.
        """
        for season in range(2,13):
            for mode in range(self.n_modes):
                if np.correlate(self.pcs[f"{season-1}"][mode], self.pcs[f"{season}"][mode]) <= 0:
                    self.pcs[f"{season}"][mode] = -self.pcs[f"{season}"][mode]
                    self.modes[f"{season}"][mode] = -self.modes[f"{season}"][mode]
    
    def filter_variables(self):
        """
        Filters the data according to the variables selected
        """
        self.data = self.data[[var.lower() for var in self.variables_dict.values()]]

    def calculate_index_by_season(self, biseason: int):
        """
        Performs PCA for a specific month and returns the PCs, explained variance, and modes.

        Parameters:
        biseason : Target month for which PCA is performed (1-12).

        Returns:
        tuple:
            - Array of principal components.
            - Array of explained variance for each mode.
            - Array of modes.
            - Boolean mask for valid columns.
        """
        if self.rolling_window > 1:
            bimonthly = self.data.rolling(time=self.rolling_window, center=True).mean() #first month is DJ
        else:
            bimonthly = self.data
        target_bimonth = bimonthly.sel(time=is_month(bimonthly['time.month'], biseason))
        target_bimonth = target_bimonth.sel(time=slice(f"{self.start_year}-01", f"{self.end_year}-12"))

        flatten_bimonth, valid_columns = flatten_vector(target_bimonth, target_bimonth.sizes["time"])

        cov_matrix = np.matmul(flatten_bimonth, flatten_bimonth.T)/(flatten_bimonth.shape[0]-1)
        N = cov_matrix.shape[0]
        D, V= sp.linalg.eigh(cov_matrix, subset_by_index=[N-self.n_modes,N-1])

        # Reverse for descending order
        D = D[::-1]
        V = V[:, ::-1]

        pcs = []
        modes = []

        for i in range(self.n_modes):
            # Project kth eigenvector onto the original space
            V_k = np.matmul(flatten_bimonth.T, V[:, i])

            # Normalize the mode
            sq = np.sqrt(D[i])
            V_k = V_k / sq

            # Compute the principal component time series
            pc_k = V_k.T @ flatten_bimonth.T / (np.dot(V_k.T, V_k))

            pcs.append(pc_k)
            modes.append(V_k)

        return np.array(pcs), D / np.trace(cov_matrix), np.array(modes), valid_columns

    def map_modes_to_full_grid(self, modes: int, valid_columns: np.ndarray):
        """
        Maps modes back to the full grid, filling invalid points with NaNs.

        Parameters:
        modes : Array containing the modes for valid columns.
        valid_columns : Boolean mask for valid columns.

        Returns: List of modes mapped to the full grid.
        """
        lat, lon = self.grid_shape
        mode_length_per_variable = lat * lon

        full_modes = []
        for mode in modes:
            full_mode = np.full((mode_length_per_variable * self.n_variables,), np.nan)  
            full_mode[valid_columns] = mode  
            full_modes.append(full_mode)

        return full_modes

    def plot_variable_modes_shared(self, season: int, figtitle=None):
        """
        Plot the spatial modes for each variable for a given season.

        Args:
            season: The season for which modes are plotted
            figtitle: Title for the figure. Defaults to None.

        This method plots each mode of spatial variability for each variable, creating a grid of subplots for 
        all modes and variables. Each subplot is scaled to a common color range, with a shared colorbar.
        """
        lat, lon = self.grid_shape
        mode_length_per_variable = lat * lon

        for i in range(self.n_modes):
            fig, axes = plt.subplots(1, self.n_variables, figsize=(20, 5), constrained_layout=True)
            if self.n_variables == 1:
                axes = [axes]

            reshaped_modes = []
            for j in range(self.n_variables):
                variable_mode = self.modes[f"{season}"][i][j * mode_length_per_variable : (j + 1) * mode_length_per_variable]
                reshaped_mode = variable_mode.reshape(self.grid_shape)
                reshaped_modes.append(reshaped_mode)

            vmin = np.nanmin(reshaped_modes)
            vmax = np.nanmax(reshaped_modes)

            im_list = []
            for j, ax in enumerate(axes):
                im = ax.imshow(reshaped_modes[j], cmap='coolwarm', origin='lower', vmin=vmin, vmax=vmax)
                ax.set_title(f"Mode {i+1} for Variable {self.variables_dict[j+1]}")
                im_list.append(im)

            # Create a single colorbar for all subplots
            if self.n_variables != 1:
                cbar = fig.colorbar(im_list[0], ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
            else:
                cbar = fig.colorbar(im_list[0], ax=axes[0], orientation='vertical', fraction=0.02, pad=0.04)
            fig.suptitle(f"{i+1}-Mode for Season {season}", fontsize=16)
            plt.show()
    
    def get_full_index(self):
        """
        Compile the full index.
        """
        self.index = {}
        for mod in range(self.n_modes):
            aux_index = []
            for i in range(len(self.pcs["1"][mod])):
                for k in range(1,13):
                    aux_index.append(self.pcs[f"{k}"][mod][i])
            self.index[f"PC-{mod+1}"] = np.array(aux_index)
    
    def get_full_index_yearly(self):
        """
        Compile the full index.
        """
        self.index = {}
        for mod in range(self.n_modes):
            self.index[f"PC-{mod+1}"] = np.array(self.pcs[mod])
    
    def get_index(self, mode: int, start_year=1980, end_year=2022):
        """
        Retrieves the full index.
        """
        if (start_year-self.start_year < 0) or (self.end_year-end_year < 0):
            print("Can not retrieve for not calculated years")
        else:
            if self.freq=="monthly":
                return self.index[f"PC-{mode}"][(start_year-self.start_year)*12:(end_year-self.start_year+1)*12]
            elif self.freq=="yearly":
                return self.index[f"PC-{mode}"][(start_year-self.start_year):(end_year-self.start_year+1)]
    
    def get_index_by_season(self, season: int, mode: int, start_year=1980, end_year=2022):
        """
        Retrieves the full index by season.
        """
        if (start_year-self.start_year < 0) or (self.end_year-end_year < 0):
            print("Can not require for not calculated years")
        else:
            return self.pcs[f"{season}"][mode-1][(start_year-self.start_year):(end_year-self.start_year+1)]
    
    def plot_explained_variance(self):
        """
        Plot the variance explained by each mode across seasons.

        This function creates a bar plot for each biseason (1 to 12), showing the explained variance
        by each mode. Each subplot has its own scaling and labels.
        """
        if self.freq=="monthly":
            fig, axs = plt.subplots(3, 4, figsize=(20,20))
            for i in range(12):
                axs.flatten()[i].bar(np.linspace(1,self.n_modes,self.n_modes), self.explained_variance[f"{i+1}"])
                axs.flatten()[i].set_ylim([0,2*np.max(self.explained_variance[f"{i+1}"])])
                axs.flatten()[i].set_title(f"Variance explained per mode for biseason {i+1}")
                axs.flatten()[i].set_xticks([i for i in range(1, self.n_modes+1)])
        elif self.freq=="yearly":
            fig, ax = plt.subplots(figsize=(10,10))
            ax.bar(np.linspace(1,self.n_modes,self.n_modes), self.explained_variance)
            ax.set_ylim([0,2*np.max(self.explained_variance)])
            ax.set_title(f"Variance explained per mode")
            ax.set_xticks([i for i in range(1, self.n_modes+1)])
        plt.show()


## PCA + DF Visualizations

def plot_hw_index(hwis_orig: pd.DataFrame, stat: list ,idx:MultivariatePCA, mode: int, start=1980, end=2022):
    """
    Plot heatwave index against principal components over a specified date range.

    Args:
        hwis_orig (pd.DataFrame): Original heatwave index dataframe.
        stat (list): List of heatwave statistics to plot.
        idx (MultivariatePCA): PCA object containing principal component indices.
        mode (int): Mode index to compare with heatwave statistics.
        start (int): Start year for the plot. Defaults to 1980.
        end (int): End year for the plot. Defaults to 2022.

    This function plots the correlation between heatwave statistics and PCA mode indices 
    over the specified date range.
    """
    fig, ax = plt.subplots(len(stat), figsize=(18,10))
    hwis = hwis_orig[(hwis_orig.index.year <= end) & (hwis_orig.index.year >= start)]

    if len(stat) > 1:
        for k in range(len(stat)):
            if idx and mode:
                    
                corr = np.corrcoef(hwis[f"{stat[k]}"],idx.get_index(mode, start_year=start, end_year=end))[0,1]

                ax[k].plot(hwis.index, hwis[f"{stat[k]}"], color="red")
                yabs_max = abs(max(ax[k].get_ylim(), key=abs))
                ax[k].set_ylim(ymin=-yabs_max, ymax=yabs_max)
                ax[k].set_title(f"{stat[k]} and PC-{mode}, Corr: {np.round(corr,2)} from {start} to {end}")

                ax2 = ax[k].twinx()
                ax2.plot(hwis.index, idx.get_index(mode, start_year=start, end_year=end), color="blue")
                yabs_max = abs(max(ax2.get_ylim(), key=abs))
                ax2.set_ylim(ymin=-yabs_max, ymax=yabs_max)
            else:
                ax[k].plot(hwis.index, hwis[f"{stat[k]}"], color="red")
                ax[k].set_title(f"{stat[k]} from {start} to {end}")

    else:
        if idx and mode:
            corr = np.corrcoef(hwis[f"{stat[0]}"],idx.get_index(mode, start_year=start, end_year=end))[0,1]

            ax.plot(hwis.index, hwis[f"{stat[0]}"], color="red")
            yabs_max = abs(max(ax.get_ylim(), key=abs))
            ax.set_ylim(ymin=-yabs_max, ymax=yabs_max)
            ax.set_title(f"{stat[0]} and PC-{mode}, Corr: {np.round(corr,2)} from {start} to {end}")
            ax2 = ax.twinx()
            ax2.plot(hwis.index, idx.get_index(mode, start_year=start, end_year=end), color="blue")
            yabs_max = abs(max(ax2.get_ylim(), key=abs))
            ax2.set_ylim(ymin=-yabs_max, ymax=yabs_max)
        else:
            ax.plot(hwis.index, hwis[f"{stat[0]}"], color="red")
            ax.set_title(f"{stat[0]} from {start} to {end}")
        

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

def plot_hw_index_by_season(hwis: pd.DataFrame, stat: str, idx:MultivariatePCA, mode: int, start=1980, end=2022):
    """
    Plot heatwave index against principal component indices by season.

    Args:
        hwis (pd.DataFrame): Heatwave index dataframe.
        stat (str): Name of the heatwave statistic to plot.
        idx (MultivariatePCA): PCA object containing principal component indices.
        mode (int): Mode index to compare with heatwave statistics.
        start (int): Start year for the plot. Defaults to 1980.
        end (int): End year for the plot. Defaults to 2022.

    This function generates seasonal subplots for heatwave statistics against PCA mode indices, 
    displaying one subplot per season.
    """
    fig, axs = plt.subplots(4,3, figsize=(22,14))

    hwis = hwis[(hwis.index.year <= end) & (hwis.index.year >= start)]

    for i in range(12):
        if idx and mode:
            hwis_month = hwis[hwis.index.month==i+1]
            corr = np.corrcoef(hwis_month[f"{stat}"], idx.get_index_by_season(i+1, mode, start_year=start, end_year=end))[0,1]
            axs.flatten()[i].plot(hwis_month.index, hwis_month[f"{stat}"], color="red")
            yabs_max = abs(max(axs.flatten()[i].get_ylim(), key=abs))
            axs.flatten()[i].set_ylim(ymin=-yabs_max, ymax=yabs_max)
            axs.flatten()[i].set_title(f"Biseason {i+1}, Corr: {np.round(corr,2)}")
            fig.suptitle(f"{stat} and PC-{mode} per season for years {start} to {end}")


            ax2 = axs.flatten()[i].twinx()
            ax2.plot(hwis_month.index, idx.get_index_by_season(i+1, mode, start_year=start, end_year=end), color="blue")
            yabs_max = abs(max(ax2.get_ylim(), key=abs))
            ax2.set_ylim(ymin=-yabs_max, ymax=yabs_max)

        else:
            hwis_month = hwis[hwis.index.month==i+1]
            axs.flatten()[i].plot(hwis_month.index, hwis_month[f"{stat}"], color="red")
            axs.flatten()[i].set_title(f"Biseason {i+1}")

            fig.suptitle(f"{stat} from {start} to {end}")


    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # otherwise the right y-label is slightly clipped
    plt.show()

def scatter_color_year(hwis_orig: pd.DataFrame, stat: list, idx, mode: int, start=1980, end=2022):
    """
    Create a scatter plot of heatwave statistics vs PCA mode indices with color indicating year.

    Args:
        hwis_orig (pd.DataFrame): Original heatwave index dataframe.
        stat (list): List of heatwave statistics to plot.
        idx (MultivariatePCA): PCA object containing principal component indices.
        mode (int): Mode index to compare with heatwave statistics.
        start (int): Start year for color range normalization. Defaults to 1980.
        end (int): End year for color range normalization. Defaults to 2022.

    This function generates a scatter plot of heatwave statistics against PCA mode indices, 
    with point color representing the year.
    """
    fig, ax = plt.subplots(len(stat), figsize=(20, 15))
    hwis = hwis_orig[(hwis_orig.index.year <= end) & (hwis_orig.index.year >= start)]
    
    norm = mcolors.Normalize(vmin=start, vmax=end)
    cmap = cm.plasma

    if len(stat) != 1:
        for k in range(len(stat)):
            sc = ax[k].scatter(
                idx.get_index(mode, start_year=start, end_year=end),
                hwis[f"{stat[k]}"],
                c=hwis.index.year,
                cmap=cmap,
                norm=norm
            )
            ax[k].set_title(f"{stat[k]} vs PC-{mode} by year")
        fig.colorbar(sc, ax=ax, orientation='vertical', label='Year')
    else:
        sc = ax.scatter(
            idx.get_index(mode, start_year=start, end_year=end),
            hwis[f"{stat[0]}"],
            c=hwis.index.year,
            cmap=cmap,
            norm=norm
        )
        fig.suptitle(f"{stat[0]} vs PC-{mode} by year")
        fig.colorbar(sc, ax=ax, orientation='vertical', label='Year')

    plt.show()

def scatter_color_season(hwis_orig: pd.DataFrame, stat: list, idx, mode: int, start=1980, end=2022):
    """
    Create a scatter plot of heatwave statistics vs PCA mode indices with color indicating season.

    Args:
        hwis_orig (pd.DataFrame): Original heatwave index dataframe.
        stat (list): List of heatwave statistics to plot.
        idx (MultivariatePCA): PCA object containing principal component indices.
        mode (int): Mode index to compare with heatwave statistics.
        start (int): Start year for the plot. Defaults to 1980.
        end (int): End year for the plot. Defaults to 2022.

    This function generates a scatter plot of heatwave statistics against PCA mode indices, 
    with point color representing the month or season.
    """
    fig, ax = plt.subplots(len(stat), figsize=(20, 15))
    hwis = hwis_orig[(hwis_orig.index.year <= end) & (hwis_orig.index.year >= start)]
    
    norm = mcolors.Normalize(vmin=1, vmax=12)
    cmap = cm.plasma

    if len(stat) != 1:
        for k in range(len(stat)):
            sc = ax[k].scatter(
                idx.get_index(mode, start_year=start, end_year=end),
                hwis[f"{stat[k]}"],
                c=hwis.index.month,
                cmap=cmap,
                norm=norm
            )
            ax[k].set_title(f"{stat[k]} vs PC-{mode} by season")
        fig.colorbar(sc, ax=ax, orientation='vertical', label='Year')
    else:
        sc = ax.scatter(
            idx.get_index(mode, start_year=start, end_year=end),
            hwis[f"{stat[0]}"],
            c=hwis.index.month,
            cmap=cmap,
            norm=norm
        )
        fig.suptitle(f"{stat[0]} vs PC-{mode} by season")
        fig.colorbar(sc, ax=ax, orientation='vertical', label='Year')

    plt.show()
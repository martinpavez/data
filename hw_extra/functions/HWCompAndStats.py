import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import numpy as np
import pandas as pd
import seaborn as sns

#from scipy import signal
#import hw_predictor.utils as su


from scipy.stats import norm
from scipy.optimize import curve_fit

from calendar import monthrange
from datetime import datetime, timedelta
from scipy import stats

from matplotlib.ticker import NullLocator



def model(x, params):
    a, a1, b1 = params
    omega = 2 * np.pi / 365.25  # 365.25
    theta = np.angle(b1 + 1j*a1) #np.arctan(a1 / b1)
    alpha = np.sqrt(a1**2 + b1**2)

    y_pred = a + alpha * np.sin(omega * x + theta)
    return y_pred

def model_fit(x, a, a1, b1):
    omega = 2 * np.pi / 365.25
    y_pred = a + a1 * np.cos(omega * x) + b1 * np.sin(omega * x)
    return y_pred

def RSS(y, y_pred) -> float:
    return np.sqrt((y - y_pred) ** 2).sum()

def compute_quartiles(row):
    quartiles = np.percentile(row, [25, 50, 75])
    return pd.Series(quartiles, index=['Q1', 'Q2', 'Q3'])

def CTpct(current_date, historical_df, length_window):
    # Extract the day and month from the current date
    day_of_year = current_date.dayofyear
    l = int(length_window/2)
    # Define a 15-day window centered on the current day
    window_start =  day_of_year - l
    window_end =    day_of_year + l

    # Filter the historical data for this 15-day window
    historical_window = historical_df[(historical_df.index.dayofyear >= window_start) & 
                                      (historical_df.index.dayofyear <= window_end)]
    
    # Calculate the 90th percentile
    hwq = historical_window.quantile(0.90)

    return hwq




#HW computing function
class HW_statistics:
    def __init__(self, data, start_year, end_year):
        super().__init__()
        self.data = data
        try:
            self.max_temp = data.set_index(data.index.normalize()).dropna(subset=["max_temp"])[["max_temp"]]
        except AttributeError: # evita un error en caso que la fecha ya esté normalizada
            self.max_temp = data.set_index(data.index).dropna(subset=["max_temp"])[["max_temp"]]
        try:
            self.mean_temp = data.set_index(data.index.normalize()).dropna(subset=["mean_temp"])[["mean_temp"]]
        except KeyError:
            ...
        try:
            self.min_temp = data.set_index(data.index.normalize()).dropna(subset=["min_temp"])[["min_temp"]]
        except KeyError:
            ...
        try:
            self.data_dict = {'max_temp': self.max_temp, 'mean_temp': self.mean_temp, 'min_temp': self.min_temp}
        except AttributeError:
            self.data_dict = {'max_temp': self.max_temp}

        self.start_year = start_year
        self.end_year = end_year

        self.start_date = datetime(self.start_year, 1, 1)
        self.end_date = datetime(self.end_year, 12, 31)

        year = 2020  ## es un año bisiesto cualquiera como referencia.
        self.month_days = self.Month_Days(year)#{"year": year, "month_days": {m: monthrange(year, m)[1] for m in range(1, 13)}}

    def Month_Days(self, year):
        return {"year": year, "month_days": {m: monthrange(year, m)[1] for m in range(1, 13)}}

    def Tind(self, Tind_type: str): #, data_temp: pd.DataFrame):
        #Tind_type : 'mean', 'max_temp', 'min_temp', etc...

        # compute the thresholds ####90-percentile for each month/day in the year
        data_Tind = pd.DataFrame(
            [],
            columns=[Tind_type], #"max_temp", "P90", "mean", "mean+5", "P90_russo_set"],
            index=pd.Index(pd.date_range(self.start_date, self.end_date, freq="D"), name="date"),
        )

        mask = (self.data_dict[Tind_type].index >= self.start_date) & (self.data_dict[Tind_type].index <= self.end_date)
        data_Tind[Tind_type] = self.data_dict[Tind_type][mask]
        return data_Tind

    def String_Trcrit(self, Tcrit_dict):
        str_tcrit_type = Tcrit_dict['meas'] + str(Tcrit_dict['perc']) + Tcrit_dict['adj'] + str(Tcrit_dict['set'])
        if Tcrit_dict['Tadd'] > 0:
            str_tcrit_type = str_tcrit_type + '+' + str(Tcrit_dict['Tadd'])
        return str_tcrit_type

    def Tcrit(self, Tcrit_dict: dict, year_window_init: int, year_window_end: int):
        #Tcrit_dict['meas'] : 'perc', 'mean', 'max_temp', 'constant', etc...
        #Tcrit_dict['perc'] : 85, 90, 95, '', ...
        #Tcrit_dict['adj'] : 'adj', '', ...
        #Tcrit_dict['set'] : 'custom', 'russo_set' ...
        #Tcrit_dict['Tadd'] : 0, 1, 3, 4, 5, ...
        #Tcrit_dict['hist_data_temp'] : 'max_temp', 'mean', etc <--- medida de los datos historicos sobre la cual se obtiene el percentil, media, etc.
        year_window_init = year_window_init
        year_window_end = year_window_end
        data_temp = self.data[Tcrit_dict['hist_data_temp']]
        str_tcrit_type = self.String_Trcrit(Tcrit_dict)

        data_threshold = pd.DataFrame(
            [],
            columns=[str_tcrit_type],#"max_temp", "P90", "mean", "mean+5", "P90_russo_set"],
            index=pd.Index(pd.date_range(self.start_date, self.end_date, freq="D"), name="date"),
        )
        if Tcrit_dict['meas'] == 'perc' and Tcrit_dict['adj'] == 'adj':
            Tcrit_dict_aux = {'meas': 'perc',
                              'perc': Tcrit_dict['perc'],
                              'adj': '', 'set': '',
                              'Tadd': Tcrit_dict['Tadd'],
                              'hist_data_temp': Tcrit_dict['hist_data_temp']}
            data_threshold_aux = self.Tcrit(Tcrit_dict_aux, year_window_init, year_window_end)
            temp_t = data_threshold_aux[self.String_Trcrit(Tcrit_dict_aux)].copy(deep=True)
            temp_t = temp_t.to_frame()
            if isinstance(temp_t.index, pd.DatetimeIndex):
                first_ord = temp_t.index.map(datetime.toordinal)[0]
                temp_t.index = temp_t.index.map(datetime.toordinal)
            params, cov = curve_fit(
            model_fit, xdata=temp_t.index - first_ord, ydata=temp_t[self.String_Trcrit(Tcrit_dict_aux)], method="lm"
            )
            param_list = ["a", "a1", "b1"]
            std_dev = np.sqrt(np.diag(cov))

            data_threshold.loc[:, str_tcrit_type] = model(temp_t.index - first_ord, params)
            if isinstance(temp_t.index, pd.DatetimeIndex):
                temp_t.index = temp_t.index.map(datetime.toordinal)
            #########################
        elif Tcrit_dict['meas'] == 'perc' and isinstance(Tcrit_dict['set'], int):
            #CTpct(current_date, historical_df, length_window)
            historical_df = data_temp[
                                (year_window_init <= data_temp.index.year)
                                & (data_temp.index.year <= year_window_end)
                            ]
            #print(historical_df.head())
            data_threshold.loc[:, str_tcrit_type] = data_temp.index.to_series().apply(lambda x: CTpct(x, 
                                                    historical_df, Tcrit_dict['set']))


        else:
            for year in range(self.start_date.year, self.end_date.year + 1):
                month_days = self.Month_Days(year)
                for month in month_days["month_days"]:
                    for day in range(1, month_days["month_days"][month] + 1):
                        try:
                            current_date = datetime(year, month, day)
                        except ValueError:
                            if current_date == self.start_date:
                                current_date = datetime(year, month, day+1)
                            if current_date == self.end_date:
                                current_date = datetime(year, month, day-1)
                        if  current_date >= self.start_date and current_date <= self.end_date:
                            f_data_temp = data_temp[
                                (year_window_init <= data_temp.index.year)
                                & (data_temp.index.year <= year_window_end)
                                & (data_temp.index.day == day)
                                & (data_temp.index.month == month)
                            ]
                            if Tcrit_dict['meas'] == 'perc':
                                try:
                                    data_threshold.loc[datetime(year, month, day), str_tcrit_type] = f_data_temp.quantile(
                                        Tcrit_dict['perc']*0.01, interpolation="midpoint"
                                    ).values[0]
                                except AttributeError:
                                    data_threshold.loc[datetime(year, month, day), str_tcrit_type] = f_data_temp.quantile(
                                        Tcrit_dict['perc']*0.01, interpolation="midpoint"
                                    )
                                except ValueError:
                                    data_threshold.loc[datetime(year, month, day-1), str_tcrit_type] = f_data_temp.quantile(
                                        Tcrit_dict['perc']*0.01, interpolation="midpoint"
                                    ).values[0] + Tcrit_dict['Tadd']
                            elif Tcrit_dict['meas'] == 'mean':
                                try:
                                    data_threshold.loc[datetime(year, month, day), str_tcrit_type] = f_data_temp.mean()
                                except ValueError:
                                    data_threshold.loc[datetime(year, month, day-1), str_tcrit_type] = f_data_temp.mean()
                            else:
                                print('threshold type still not supported...')
                                return []
        return data_threshold + Tcrit_dict['Tadd']
    def HW_funs(self, Tind_type: str,
        Tcrit_dict: dict,
        Nd: int,
        year_window_init: int, year_window_end: int):#,

        data_Tind = self.Tind(Tind_type)
        data_threshold = self.Tcrit(Tcrit_dict, year_window_init, year_window_end)
        Tcrit_type = self.String_Trcrit(Tcrit_dict)
        # create binary column with 1 if the max_temp is above the (personal) threshold
        data_threshold.loc[:, 'Tind'] = data_Tind.copy(deep=True) #### quitar (?)
        data_threshold.loc[:, "above_threshold"] = (
            data_threshold[Tcrit_type] < data_Tind[Tind_type]
        ).astype(int)

        temps_above_threshold = data_threshold[data_threshold["above_threshold"] == 1].copy(deep=True)

        # create dataframe with heatwaves information (start, end, duration, mean_temp, max_temp, date)
        heatwaves = pd.DataFrame(
            columns=["start", "end", "duration", "mean_temp", "min_temp", "max_temp", "date_max", "magnitude", "max_magn"]
        )

        # get start and end of continuous heatwaves
        heatwaves_dates = []  # [(start, end), (start, end), ...]
        for i in range(1, len(temps_above_threshold)):
            # continuous heatwaves
            if temps_above_threshold.index[i] - temps_above_threshold.index[i - 1] == timedelta(days=1):
                # if empty, add first heatwave
                if len(heatwaves_dates) == 0:
                    heatwaves_dates.append(
                        (temps_above_threshold.index[i - 1], temps_above_threshold.index[i])
                    )
                # if not empty, check if the last heatwave is continuous
                else:
                    # if continuous, update end date
                    if heatwaves_dates[-1][1] == temps_above_threshold.index[i - 1]:
                        heatwaves_dates[-1] = (heatwaves_dates[-1][0], temps_above_threshold.index[i])
                    # if not continuous, add new heatwave
                    else:
                        heatwaves_dates.append(
                            (temps_above_threshold.index[i - 1], temps_above_threshold.index[i])
                        )

        for idx, hw_dates in enumerate(heatwaves_dates):
            # get start and end dates
            start = hw_dates[0]
            end = hw_dates[1]

            # get duration
            duration = (end - start).days + 1

            # get mean and max temp
            mean_temp = self.data.loc[start:end, Tind_type].mean()
            min_temp = self.data.loc[start:end, Tind_type].min()
            max_temp = self.data.loc[start:end, Tind_type].max()
            magnitude = (self.data.loc[start:end, Tind_type] - data_threshold.loc[start:end, Tcrit_type]).mean()
            # get max_temp date
            date = self.data.loc[start:end, Tind_type].idxmax()
            max_magn = (self.data.loc[date, Tind_type] - data_threshold.loc[date, Tcrit_type]).max()
            heatwaves.loc[idx] = [start, end, duration, mean_temp, min_temp, max_temp, date, magnitude, max_magn]

        # filter heatwaves with duration >= Nd days
        heatwaves = (
            heatwaves[heatwaves["duration"] >=  Nd]
            .sort_values(by="start", ascending=False)
            .reset_index(drop=True)
        )

        return data_Tind, data_threshold, heatwaves#, heatwave_idx
##### now the plots!!!


    def plot_single_heatwave(self,
                             Tind_type: str,
                             Tcrit_dict: dict,
                             Nd: int,
                             start_year_plot: int,
                             end_year_plot: int,
                             #year_window_init: int,
                             #year_window_end: int,
                             data_Tind: pd.DataFrame,
                             data_threshold: pd.DataFrame,
                             heatwaves: pd.DataFrame, 
                             include_months = False,
                             fsx = 10, 
                             fsy = 4.3, 
                             Tcrit_label=None, 
                             Tind_label=None, 
                             save = False,
                             filename = None, 
                             data_Tind_marker = '.-', 
                             s_Tind = 5):
        sns.set_theme(rc={'figure.figsize':(20,8.27)})
        #sns.set_theme(style="darkgrid")
        sns.set_theme(style="ticks")
        sns.set_palette("pastel")
        
        Tcrit_type = self.String_Trcrit(Tcrit_dict)
        data_threshold = data_threshold[data_threshold.index.year >= start_year_plot]
        data_threshold = data_threshold[data_threshold.index.year <= end_year_plot]
        data_Tind = data_Tind[data_Tind.index.year >= start_year_plot]
        data_Tind = data_Tind[data_Tind.index.year <= end_year_plot]
        heatwaves = heatwaves[heatwaves['start'].dt.year >= start_year_plot]
        heatwaves = heatwaves[heatwaves['start'].dt.year <= end_year_plot] #las olas de calor deben comenzar en el periodo establecido
        
        if Tcrit_label is None:
            Tcrit_label = Tcrit_type
        if Tind_label is None:
            Tind_label = Tind_type
        fig = plt.figure(figsize=(fsx, fsy))
        sns.lineplot(
            data = data_threshold,
            y=Tcrit_type,
            x="date",
            label=Tcrit_label,
            lw=3,
        )

        sns.scatterplot(
            data = data_Tind,
            y=Tind_type,
            x="date",
            #label=Tind_label,
            color="red",
            s=s_Tind,
        )

        plt.plot(data_Tind, data_Tind_marker, color='red', linewidth=1, label=Tind_label)

        #plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.legend(bbox_to_anchor=(0.01, 0.21), loc='upper left')#, alpha=1)

        # vertical span for heatwaves
        heatwaves_dates = heatwaves[["start", "end"]].values
        for hw_dates in heatwaves_dates:
            plt.axvspan(hw_dates[0], hw_dates[1], facecolor="#DE4D4D", alpha=0.3)

        plt.xlim([data_Tind.index.min(), data_Tind.index.max()])
        #plt.xticks(rotation=45)

        if include_months:
            # Format the x-axis to show months
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # Major ticks: Months
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # Format months

            # Hide minor tick marks on the main x-axis
            plt.gca().tick_params(axis='x', which='minor', length=0)

            # Add a secondary x-axis for the year labels
            secax = plt.gca().secondary_xaxis('bottom')
            secax.xaxis.set_minor_locator(mdates.YearLocator())
            secax.xaxis.set_minor_formatter(mdates.DateFormatter('%Y'))

            # Hide major ticks on the secondary x-axis
            secax.tick_params(axis='x', which='major', length=0)

            # Set the label visibility and remove the actual tick lines for the secondary axis
            for label in secax.get_xticklabels(which='major'):
                label.set_visible(False)
            for tick in secax.xaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)

            # Adjust padding to position years below months
            secax.tick_params(axis='x', which='minor', pad=28, length=10, width=1.5, labelrotation = 0, labelright=True, labelsize=12)

            # Rotate month labels for better readability
            plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45, ha='left')#, va='top')

        else:
        #if self.end_date.year - self.start_date.year >= Nd:
            plt.gca().xaxis.set_major_locator(mdates.YearLocator())
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

            plt.xticks(rotation=45)

        plt.xlabel('')#"Date")
        plt.ylabel("Temperature [Cº]")
        #plt.title("Heatwaves:" + ' (' + Tind_type + ', '\
        #            + Tcrit_type + '_tw: [' + str(year_window_init) + ',' + str(year_window_end) + '], ' + str(Nd) + ')')
        #+ r"$(T_{max}, \underset{over \ years}{Mean}(T_{max}),$" + str(Nd) + "$)$")
        plt.show()

        if save:
            fig.savefig(filename, format='pdf', bbox_inches='tight')





def HW_indices_by_year(hw1, HWDef, start_year, end_year, year_window):
    #this function is for a single HW definition and for a single station
    hw_stats_by_year = {}
    for yw in list(year_window.keys()): #range(1, len(year_window)+1):
        hw_annual_stats = {}
        _, _, heatwaves = hw1.HW_funs(HWDef['Tind_type'], HWDef['Tcrit_dict'], HWDef['Nd'], year_window[yw][0], year_window[yw][1])
        hw_annual_stats = pd.DataFrame(columns=['HWN', 'HWF', 'HWD', 'HWM', 'HWA'], index=range(start_year, end_year+1))
        for year_to_analyze in range(start_year, end_year+1):
            hw_year = heatwaves[heatwaves['start'].dt.year == year_to_analyze]
            hw_annual_stats['HWN'][year_to_analyze] = len(hw_year)
            hw_annual_stats['HWF'][year_to_analyze] = np.sum(hw_year['duration'])
            hw_annual_stats['HWD'][year_to_analyze] = np.max(hw_year['duration'])
            if np.isnan(hw_annual_stats['HWD'][year_to_analyze]): #if a year does not have any heatwave event, then the duration is set to be 0. 
                hw_annual_stats['HWD'][year_to_analyze] = 0.0
            hw_annual_stats['HWM'][year_to_analyze] = np.sum(np.array(hw_year['duration'], dtype=float)*np.array(hw_year['magnitude'], dtype=float))/np.sum(np.array(hw_year['duration'], dtype=float))
            if np.isnan(hw_annual_stats['HWM'][year_to_analyze]): #if a year does not have any heatwave event, then the duration is set to be 0. 
                hw_annual_stats['HWM'][year_to_analyze] = 0.0
            hw_annual_stats['HWA'][year_to_analyze] = np.max(hw_year['max_magn'])
            if np.isnan(hw_annual_stats['HWA'][year_to_analyze]): #if a year does not have any heatwave event, then the duration is set to be 0. 
                hw_annual_stats['HWA'][year_to_analyze] = 0.0
        hw_stats_by_year[yw] = hw_annual_stats

    return hw_stats_by_year


def HW_indices_summary(HWDef_dict, HW_indices, ref_period, stations, start_year, end_year, stations_data):
    #this function uses HW_stats_by_year for different definitions of heatwaves and for different stations
    hw_region_summary = {}
    for hwdef in list(HWDef_dict.keys()):
        hw_region_summary[hwdef] = {}
        for yw in list(ref_period.keys()):
            hw_region_summary[hwdef][yw] = {}
            for hwi in HW_indices:
                hw_region_summary[hwdef][yw][hwi] = pd.DataFrame(columns=['Years'] +stations.index.tolist())#["station_id", "min_year", "max_year"])
                hw_region_summary[hwdef][yw][hwi]['Years'] = range(start_year, end_year+1)
                hw_region_summary[hwdef][yw][hwi].set_index('Years', inplace = True)

    for station in stations.index.tolist():
        station_id = str(station)
        data = stations_data[station].copy()
        for hwdef in list(HWDef_dict.keys()):
            print('station=' + station_id)
            print('hwdef=')
            print(hwdef)
            hw1 = HW_statistics(data, start_year, end_year)
            hw_stats_by_year = HW_indices_by_year(hw1, HWDef_dict[hwdef], start_year, end_year, ref_period)
            for yw in list(ref_period.keys()):#range(1, len(year_window)+1):
                for hwi in HW_indices:
                    hw_region_summary[hwdef][yw][hwi][station] = hw_stats_by_year[yw][hwi]

    return hw_region_summary



def linear_fit(x, a, b):
    return a*x + b


def HW_stats_of_indices(hw_region_summary_dict):
# this function computes the mean and std or the quartiles of the heatwave indices
    hwi_stats_region = {}
    for hwdef in list(hw_region_summary_dict.keys()):
        hwi_stats_region[hwdef] = {}
        for yw in list(hw_region_summary_dict[hwdef].keys()):
            hwi_stats_region[hwdef][yw] = {}
            for hwi in list(hw_region_summary_dict[hwdef][yw].keys()):
                hwi_stats_region[hwdef][yw][hwi] = pd.DataFrame(columns = ["mean", "std", 'min', 'max', 'Q1', 'Q2', 'Q3'])#, "median"])
                hwi_stats_region[hwdef][yw][hwi]["mean"] = np.mean(hw_region_summary_dict[hwdef][yw][hwi], axis = 1)
                hwi_stats_region[hwdef][yw][hwi]["std"] = np.std(hw_region_summary_dict[hwdef][yw][hwi], axis = 1)
                hwi_stats_region[hwdef][yw][hwi]["min"] = np.min(hw_region_summary_dict[hwdef][yw][hwi], axis = 1)
                hwi_stats_region[hwdef][yw][hwi]["max"] = np.max(hw_region_summary_dict[hwdef][yw][hwi], axis = 1)
                quartiles_hw_region = hw_region_summary_dict[hwdef][yw][hwi].apply(compute_quartiles, axis=1)

                hwi_stats_region[hwdef][yw][hwi]['Q1'] = quartiles_hw_region['Q1'].copy()#, 'Q2', 'Q3'] = quartiles_hw_CC
                hwi_stats_region[hwdef][yw][hwi]['Q2'] = quartiles_hw_region['Q2'].copy()#, 'Q2', 'Q3'] = quartiles_hw_CC
                hwi_stats_region[hwdef][yw][hwi]['Q3'] = quartiles_hw_region['Q3'].copy()#, 'Q2', 'Q3'] = quartiles_hw_CC
    return hwi_stats_region


def plot_stats_of_hwi(hwi_stats_region: dict, indice: str, ref_period:str, stat = 'Q', add_line = False,
                      add_slope = False,
                      saveplot = False, folder = None, filename = 'hwdef_and_region.pdf',
                      ymax = None):
    sns.set_theme(rc={'figure.figsize':(20,8.27)})
    #sns.set_theme(style="darkgrid")
    sns.set_theme(style="ticks")
    sns.set_palette("pastel")

    num_shades = 10  # Number of different shades
    colors = [f'#{i:02x}{i:02x}{i:02x}' for i in range(0, 256, 256 // (num_shades - 1))]
    fig = plt.figure(figsize=(6, 3.4))
    x = np.array(hwi_stats_region[ref_period][indice].index, dtype = float)
    if stat == 'Q':
        y = np.array(hwi_stats_region[ref_period][indice]["Q2"], dtype = float)
        y_min = np.array(hwi_stats_region[ref_period][indice]["min"], dtype = float)
        y_max = np.array(hwi_stats_region[ref_period][indice]["max"], dtype = float)
        y_q1 = np.array(hwi_stats_region[ref_period][indice]["Q1"], dtype = float)
        y_q2 = np.array(hwi_stats_region[ref_period][indice]["Q2"], dtype = float)
        y_q3 = np.array(hwi_stats_region[ref_period][indice]["Q3"], dtype = float)

        plt.plot(x, y, '-', color = colors[4], label='Median', zorder=3)
        plt.fill_between(x, y_min, y_q1, color=colors[8], alpha=1.0, label='Quartile-to-extreme Area', zorder=2)
        #plt.fill_between(x, y_q1, y_q3, color=colors[7], alpha=1.0, label='Interquartile Area', zorder=2)
        plt.fill_between(x, y_q1, y_q3, color=colors[7], alpha=1.0, label='Interquartile Area', zorder=2)
        plt.fill_between(x, y_q3, y_max, color=colors[8], alpha=1.0, zorder=2)#, label='Interquartile Area')

    elif stat == 'mean' or stat is None:
        y = np.array(hwi_stats_region[ref_period][indice]["mean"], dtype = float)
        y_min = (y - np.array(hwi_stats_region[ref_period][indice]["std"], dtype = float))
        y_max = (y + np.array(hwi_stats_region[ref_period][indice]["std"], dtype = float))

        if stat is None:
            plt.plot(x, y, '-', color = colors[4], zorder=3)
        else:
            plt.plot(x, y, '-', color = colors[4], label='Mean', zorder=3)
            plt.fill_between(x, y_min, y_max, color=colors[8], alpha=1.0, label=r'$\mu \pm \sigma$ area', zorder=2)

    if add_line:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        plt.plot(x, slope*x + intercept, color='blue', label='Linear fit', alpha=0.5, zorder=2)
        print(f'slope={slope}, intercept={intercept}, r={r_value}, r2={r_value**2}')
        print(f'p={p_value}, std={std_err}')
        if add_slope:
            if p_value < 0.05:
                sign_str = f'$^*$'
            else:
                sign_str = ''
            plt.text(0.31, 0.94, f'Slope = {slope:.2f}' + sign_str + f'\n$R^2$ = {r_value**2:.2f}',
                    transform=plt.gca().transAxes, fontsize=8, verticalalignment='top',
                    bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='square,pad=0.3'))

    if stat is not None:
        plt.legend(loc = 'upper left')
 
    if ymax is None:
        if stat == 'Q':
            ymax = np.max(np.array(hwi_stats_region[ref_period][indice]["Q3"], dtype = float))+2
        elif stat == 'mean':
            ymax = np.max(np.array(hwi_stats_region[ref_period][indice]["mean"], dtype = float) \
            + np.array(hwi_stats_region[ref_period][indice]["std"], dtype = float)) +1.5 
    plt.ylim([0, ymax])
    plt.ylabel(indice)
    
    ticks = [value for value in x if value % 10 == 0]
    plt.xticks(ticks)
    
    plt.grid(True, color='white', linestyle = '-', zorder=1)
    plt.gca().set_facecolor((244/255, 244/255, 255/255))

    if saveplot:
        if stat is None:
            fig.savefig(folder + indice + '_' + filename, format='pdf')
        else:
            fig.savefig(folder + indice + '_' + str(stat) + '_' + filename, format='pdf')



def plot_t_and_u_test_p_values(hw_stats_by_year_ref, indice, yw = [1970, 2023], length = 10, t_or_u_plot = None,
                          saveplot = False, folder = None, region_name = None):
    sns.set_theme(rc={'figure.figsize':(20,8.27)})
    #sns.set_theme(style="darkgrid")
    sns.set_theme(style="ticks")
    sns.set_palette("tab10")

    yws, ywe = yw[0], yw[1]
    t_stats = np.zeros((len(range(yws+length, ywe-length)),))
    p_values = np.zeros((len(range(yws+length, ywe-length)),))
    u_stats_m = np.zeros((len(range(yws+length, ywe-length)),))
    p_values_m = np.zeros((len(range(yws+length, ywe-length)),))
    for j in range(yws+length, ywe-length):
        x = np.array([hw_stats_by_year_ref[indice]['mean'][i] for i in range(j-length,j)], dtype=float)
        y = np.array([hw_stats_by_year_ref[indice]['mean'][i] for i in range(j, j+length)], dtype=float)
        t_stats[j-(yws+length)], p_values[j-(yws+length)] = stats.ttest_ind(x, y)
        u_stats_m[j-(yws+length)], p_values_m[j-(yws+length)] = stats.mannwhitneyu(x, y, alternative='two-sided')

    if t_or_u_plot == 't_stat':
        y1 = t_stats
        y2 = p_values

        # Create the main figure and axis
        fig, ax1 = plt.subplots(figsize=(6,3.4))

        # Plot the first curve
        ax1.plot(np.array(range(yws+length, ywe-length), dtype=int), np.abs(y1), 'b-', label='t-statistic for ' + indice)
        #ax1.set_xlabel('X-axis')
        ax1.set_ylabel('t-statistic for ' + indice, color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        # Create a second y-axis that shares the same x-axis
        ax2 = ax1.twinx()

        # Plot the second curve
        ax2.plot(np.array(range(yws+length, ywe-length), dtype=int), y2, 'r-', label='p-values')
        ax2.axhline(y = 0.05, color = 'lightgray', linestyle = '--')

        ax2.set_ylabel('p-values', color='r')
        ax2.set_yscale('log')
        ax2.tick_params(axis='y', labelcolor='r')

        plt.show()
    elif t_or_u_plot == 'u_stat':
        y1 = u_stats_m
        y2 = p_values_m

        # Create the main figure and axis
        fig, ax1 = plt.subplots(figsize=(6,3.4))

        # Plot the first curve
        ax1.plot(np.array(range(yws+length, ywe-length), dtype=int), np.abs(y1), 'b-', label='u-statistic for ' + indice)
        #ax1.set_xlabel('X-axis')
        ax1.set_ylabel('u-statistic for ' + indice, color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        # Create a second y-axis that shares the same x-axis
        ax2 = ax1.twinx()

        # Plot the second curve
        ax2.plot(np.array(range(yws+length, ywe-length), dtype=int), y2, 'r-', label='p-values')
        ax2.axhline(y = 0.05, color = 'lightgray', linestyle = '--')

        ax2.set_ylabel('p-values', color='r')
        ax2.set_yscale('log')
        ax2.tick_params(axis='y', labelcolor='r')

        plt.show()

    else:
        fig = plt.figure(figsize=(6,3.4))
        plt.plot(np.array(range(yws+length, ywe-length), dtype=int), p_values, '-')#, legend= legend_t)
        plt.plot(np.array(range(yws+length, ywe-length), dtype=int), p_values_m, '-')
        plt.axhline(y = 0.05, color = 'lightgray', linestyle = '--')

        plt.ylabel('p-value for ' + indice )
        plt.yscale('log')
        plt.grid()
        
        #plt.title(r"t- and u-test for $[x-" + str(length) + ", x-1]$ and $[x, x+" + str(length-1) + "]$ intervals")
        plt.legend([r"t-test", # for $[x-" + str(length) + ", x-1]$ and $[x, x+" + str(length-1) + "]$ intervals",
                    r"u-test",
                    "Sign. level"])# for $[x-" + str(length) + ", x-1]$ and $[x, x+" + str(length-1) + "]$ intervals"])

    if saveplot:
        if t_or_u_plot == 't_stat':
            fig.savefig(folder + 't_test_' + indice + '_' + region_name + '_length=' + str(length) + '.pdf',
                        bbox_inches='tight', format='pdf')
        elif t_or_u_plot == 'u_stat':
            fig.savefig(folder + 'u_test_' + indice + '_' + region_name + '_length=' + str(length) + '.pdf',
                        bbox_inches='tight', format='pdf')
        else:
            fig.savefig(folder + 't_and_u_test_' + indice + '_' + region_name + '_length=' + str(length) + '.pdf',
                        bbox_inches='tight', format='pdf')




def plot_t_and_u_test_all_indices(hw_stats_by_year_ref, indices, meas ='mean', yw = [1971, 2023], length = 10, t_or_u_plot = None,
                          saveplot = False, folder = None, region_name = None):
    sns.set_theme(rc={'figure.figsize':(20,8.27)})
    sns.set_theme(style="ticks")
    sns.set_palette("tab10")

    yws, ywe = yw[0], yw[1]
    
    fig, ax1 = plt.subplots(figsize=(6,3.4))

    for indice in indices:
        t_stats = np.zeros((len(range(yws+length, ywe-length+2)),))
        p_values = np.zeros((len(range(yws+length, ywe-length+2)),))
        u_stats_m = np.zeros((len(range(yws+length, ywe-length+2)),))
        p_values_m = np.zeros((len(range(yws+length, ywe-length+2)),))
        for j in range(yws+length, ywe-length+2):
            x = np.array([hw_stats_by_year_ref[indice][meas][i] for i in range(j-length,j)], dtype=float)
            y = np.array([hw_stats_by_year_ref[indice][meas][i] for i in range(j, j+length)], dtype=float)
            t_stats[j-(yws+length)], p_values[j-(yws+length)] = stats.ttest_ind(x, y)
            u_stats_m[j-(yws+length)], p_values_m[j-(yws+length)] = stats.mannwhitneyu(x, y, alternative='two-sided')

        y1 = u_stats_m
        y2 = p_values_m

        ax1.plot(np.array(range(yws+length, ywe-length+2), dtype=int), y2, '-', label=indice)
        
    ax1.axhline(y = 0.05, color = 'lightgray', linestyle = '--', label='Sign.level')

    plt.ylabel('p-values')
    plt.yscale('log')
    plt.grid()
    
    plt.xlabel('x (year)')
    plt.legend(fontsize=8)
    xforticks = np.array(range(yws+length, ywe-length+2), dtype=int)
    xforticks = np.sort(np.append(xforticks, [xforticks[0]-1, xforticks[-1]+1]))
    ax1.set_xlim([xforticks[0], xforticks[-1]+0.5])
    ticks = [value for value in xforticks if value % 5 == 0]
    ax1.set_xticks(ticks)

    if saveplot:
        fig.savefig(folder + 'u_test_all_indices_' + meas + '_' + region_name + '_length=' + str(length) + '.pdf',
                        bbox_inches='tight', format='pdf')


def plot_kolmog_or_levene_test_all_indices(hw_stats_by_year_ref, indices, meas ='mean', yw = [1971, 2023], length = 10, k_or_l_plot = 'kolmog',
                          saveplot = False, folder = None, region_name = None):
    sns.set_theme(rc={'figure.figsize':(20,8.27)})
    sns.set_theme(style="ticks")
    sns.set_palette("tab10")

    yws, ywe = yw[0], yw[1]
    
    fig, ax1 = plt.subplots(figsize=(6,3.4))

    for indice in indices:
        u_stats_m = np.zeros((len(range(yws+length, ywe-length+2)),))
        p_values_m = np.zeros((len(range(yws+length, ywe-length+2)),))
        for j in range(yws+length, ywe-length+2):
            x = np.array([hw_stats_by_year_ref[indice][meas][i] for i in range(j-length,j)], dtype=float)
            y = np.array([hw_stats_by_year_ref[indice][meas][i] for i in range(j, j+length)], dtype=float)
            if k_or_l_plot == 'kolmog':
                u_stats_m[j-(yws+length)], p_values_m[j-(yws+length)] = stats.ks_2samp(x, y) #stats.mannwhitneyu(x, y, alternative='two-sided')
            elif k_or_l_plot == 'levene':
                u_stats_m[j-(yws+length)], p_values_m[j-(yws+length)] = stats.levene(x, y) #stats.mannwhitneyu(x, y, alternative='two-sided')

        y1 = u_stats_m
        y2 = p_values_m

        ax1.plot(np.array(range(yws+length, ywe-length+2), dtype=int), y2, '-', label=indice)
        
    ax1.axhline(y = 0.05, color = 'lightgray', linestyle = '--', label='Sign.level')

    plt.ylabel('p-values')#+ ' in Central Chile')
    plt.yscale('log')
    #plt.xticks(rotation=45)
    plt.grid()
    
    plt.xlabel('x (year)')
    plt.legend(fontsize=8)

    #fig_PATH = 'notebooks/hcarrillo'
    if saveplot:
        fig.savefig(folder + k_or_l_plot + '_test_all_indices_' + meas + '_' + region_name + '_length=' + str(length) + '.pdf',
                        bbox_inches='tight', format='pdf')
        




def get_trends(start, end, df):
    #Filtrar los datos según el rango de años
    df_start=df[df.index>=start]
    df_end=df_start[df_start.index<=end]
    usable=df_end.columns.values.tolist()

    #Descartar las columnas con valores nulos
    for item in df_end.columns.values:
        if df_end[item].isna().values.any():
            usable.remove(item)
    df_final=df_end[usable]

    #Calcular las tendencias y significancia
    trends={}
    r_value={}
    p_values = {}

    for item in usable:
        X=np.array(df_final.index, dtype = float)
        Y=np.array(df_final[item].values, dtype = float)

        trends[item], intercept, r_value[item], p_values[item], std_err = stats.linregress(X, Y)

    return trends, r_value, p_values



#### indicatrix of a heatwave, with daily frequency.
def HW_idx(heatwaves: pd.DataFrame,
        start_date_idx: pd.Timestamp,
        end_date_idx: pd.Timestamp,
        freq='D'): #daily, by default
    range_idx = pd.date_range(start=start_date_idx, end=end_date_idx, freq=freq).tolist()
    heatwaves_aux = heatwaves[heatwaves['start'].dt.year >=start_date_idx.year]
    heatwaves_aux = heatwaves_aux[heatwaves_aux['start'].dt.year <=end_date_idx.year]
    hw_idx = pd.DataFrame(np.zeros((len(range_idx),)), columns=['HW_idx'], index=range_idx)
    for i in range(len(heatwaves_aux)):
        sd = heatwaves_aux['start'].iloc[i]
        ed = heatwaves_aux['end'].iloc[i]
        hw_idx.loc[sd:ed, 'HW_idx'] = 1
    return hw_idx





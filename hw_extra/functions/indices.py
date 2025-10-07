# heatwaves/analysis/indices.py

import pandas as pd
import numpy as np


INDICES = ["hwn", "hwf", "hwd", "hwm", "hwa", "hwmeand", "hwi", "hwmaxi", "hwmeani"]

def transform_time_label(df, new_time):
    columns = df.columns
    df = df.sort_index()
    if new_time == "year":
        df[new_time] = df.index.year
    elif new_time == "decade":
        df[new_time] = (df.index.year // 10) * 10
    else:
        raise ValueError(f"{new_time} not implemented")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for drop_col in ["time", "Date"]:
        if drop_col in numeric_cols:
            numeric_cols.remove(drop_col)

    agg_dict = {}
    for col in numeric_cols:
        if col in ['hwn', 'hwf', 'hwi']:
            agg_dict[col] = 'sum'
        elif col in ['hwd', 'hwa', 'hwmaxi']:
            agg_dict[col] = 'max'
        # skip hwm, hwmd for custom handling

    # Aggregate yearly/decadal without special indices
    df_grouped = df.groupby(new_time)[numeric_cols].agg(agg_dict)

    # --- Custom aggregations ---

    # hwm: weighted mean by hwf
    def weighted_mean_hwm(group):
        if group["hwf"].sum() == 0:
            return 0.0
        return (group["hwm"] * group["hwf"]).sum() / group["hwf"].sum()

    # hwmeand: weighted mean by number of events (hwn)
    def weighted_mean_hwmeand(group):
        if group["hwn"].sum() == 0:
            return 0.0
        return (group["hwmeand"] * group["hwn"]).sum() / group["hwn"].sum()
    
    # hwmeani: weighted mean by number of events (hwn)
    def weighted_mean_hwmeani(group):
        if group["hwn"].sum() == 0:
            return 0.0
        return (group["hwmeani"] * group["hwn"]).sum() / group["hwn"].sum()

    # Compute for years
    if "hwm" in df.columns:
        hwm_group = df.groupby(new_time).apply(weighted_mean_hwm)
        df_grouped["hwm"] = hwm_group
    if "hwmeand" in df.columns:
        hwmeand_group = df.groupby(new_time).apply(weighted_mean_hwmeand)
        df_grouped["hwmeand"] = hwmeand_group
    if "hwmeani" in df.columns:
        hwmeani_group = df.groupby(new_time).apply(weighted_mean_hwmeani)
        df_grouped["hwmeani"] = hwmeani_group

    if new_time == "year":
        df_grouped["time"] = pd.to_datetime(df_grouped.index, format='%Y')
        df_grouped.set_index("time", inplace=True)

    return df_grouped[columns]

def get_heatwaves_indices(
    hw: pd.DataFrame,
    target_years: tuple[int, int],
    kind: str = "all",
    date_col: str = "start",
    time_scale: str = "month",
    **kwargs,
):
    """
    Compute some heatwaves indices for better understanding of the heatwaves.
    The input dataframes are the output of the `detect_heatwaves` function.
    Kind of indices:
    - hwn: Number of heatwaves for the time_scale period.
    - hwf: Total duration of heatwaves for the time_scale period.
    - hwd: Maximum duration of heatwaves for the time_scale period.
    - hwm: Mean anomalies of heatwaves for the time_scale period.
    - hwa: Maximum anomalies of heatwaves for the time_scale period.
    - hwmeand: Mean duration of heatwaves for the time_scale period.
    - hwi: Total intensity of heatwaves for the time_scale period.
    - hwmaxi: Maximum intensity of heatwaves for the time_scale period.
    - hwmeani: Mean intensity of heatwaves for the time_scale_period

    Parameters
    ----------
    hw : pd.DataFrame
        Heatwaves dataframe from `detect_heatwaves` function.
    target_years: tuple
        Range of the years for computation (start_year, end_year).
    kind : str, optional
        The kind of index to compute, by default "all".
        If "all", all indices are computed and returned.
    date_col : str, optional
        The column name to use as date, by default "start".
    time_scale:: str, optional
        The time scale for heatwave indices computation, by default "month".
        Options are "month", "year", "decade".

    Returns
    -------
    pd.DataFrame
        A DataFrame built with the concatenated indices on the period.
    """
    indices = list()
    if isinstance(kind, str):
        if kind.lower() == "all":
            kind = INDICES
        elif kind.lower() not in INDICES:
            raise ValueError(f"Unknown index: {kind}")
        else:
            kind = [kind.lower()]
    if isinstance(kind, list):
        if len(kind) == 0:
            raise ValueError("No indices specified")

        # Create complete time index for the target years
        complete_time_index = _create_complete_time_index(target_years, time_scale)

        hw = _get_timestep(hw, time_scale, date_col)
        for k in kind:
            if k.lower() not in INDICES:
                raise ValueError(f"Unknown index: {k}")
            single_index = get_single_index(hw, time_scale, kind=k)
            indices.append(single_index.reindex(columns=complete_time_index, fill_value=0))
    else:
        raise NotImplementedError(f"Unknown kind: {kind}")
    return pd.concat(indices).T.sort_index().fillna(0)


def get_single_index(hw: pd.DataFrame, time_scale: str, kind: str):
    """
    Compute a single heatwaves index.
    The input dataframes are the output of the `detect_heatwaves` function.

    Parameters
    ----------
    hw : pd.DataFrame
        Heatwaves dataframe from `detect_heatwaves` function.
    time_scale : str
        The time scale for computation.
    kind : str
        The kind of index to compute. It can be one of the following:
        - hwn: Number of heatwaves for the time_scale period.
        - hwf: Total duration of heatwaves for the time_scale period.
        - hwd: Maximum duration of heatwaves for the time_scale period.
        - hwm: Mean anomalies of heatwaves for the time_scale period.
        - hwa: Maximum anomalies of heatwaves for the time_scale period.
        - hwmeand: Mean duration of heatwaves for the time_scale period.
        - hwi: Total intensity of heatwaves for the time_scale period.
        - hwmaxi: Maximum intensity of heatwaves for the time_scale period.
        - hwmeani: Mean intensity of heatwaves for the time_scale_period

    Returns
    -------
    pd.DataFrame
        The computed index as a DataFrame.
    """
    kind = kind.lower()
    if kind == "hwn":
        result = get_hwn(hw, time_scale)
    elif kind == "hwf":
        result = get_hwf(hw, time_scale)
    elif kind == "hwd":
        result = get_hwd(hw, time_scale)
    elif kind == "hwm":
        result = get_hwm(hw, time_scale)
    elif kind == "hwa":
        result = get_hwa(hw, time_scale)
    elif kind == "hwmeand":
        result = get_hwmeand(hw, time_scale)
    elif kind == "hwi":
        result = get_hwi(hw, time_scale)
    elif kind == "hwmaxi":
        result = get_hwmaxi(hw, time_scale)
    elif kind == "hwmeani":
        result = get_hwmeani(hw, time_scale)
    else:
        raise ValueError(f"Unknown index: {kind}")

    return result


def get_hwn(hw: pd.DataFrame, time_scale: str):
    """HWN: Number of heatwaves on time_scale period."""
    return hw[time_scale].value_counts().sort_index().to_frame().rename(columns={"count": "hwn"}).T


def get_hwf(hw: pd.DataFrame, time_scale: str):
    """HWF: Total duration of heatwaves on time_scale period."""
    return hw.groupby(time_scale)["duration"].sum().to_frame().rename(columns={"duration": "hwf"}).T


def get_hwd(hw: pd.DataFrame, time_scale: str):
    """HWD: Maximum duration of heatwaves on time_scale period."""
    return hw.groupby(time_scale)["duration"].max().to_frame().rename(columns={"duration": "hwd"}).T


def get_hwm(hw: pd.DataFrame, time_scale: str):
    """
    HWM: Mean anomalies of heatwaves on time_scale period.
    Anomaly is the difference between the temperature and the threshold.
    """
    hw["sum_anomalies"] = hw["duration"] * hw["mean_anomaly"]
    hwm = hw.groupby(time_scale)["sum_anomalies"].sum() / hw.groupby(time_scale)["duration"].sum()
    hw.drop(columns="sum_anomalies", inplace=True)
    return hwm.to_frame().rename(columns={0: "hwm"}).T


def get_hwa(hw: pd.DataFrame, time_scale: str):
    """
    HWA: Amplitude of heatwaves on time_scale period.
    Amplitude is the maximum difference between the temperature and the threshold (anomaly).
    """
    return (
        hw.groupby(time_scale)["max_anomaly"]
        .max()
        .to_frame()
        .rename(columns={"max_anomaly": "hwa"})
        .T
    )


def get_hwmeand(hw: pd.DataFrame, time_scale: str):
    """
    HWMEAND: Mean Duration of heatwaves for the time_scale period.
    """
    return (
        hw.groupby(time_scale)["duration"]
        .mean()
        .to_frame()
        .rename(columns={"duration": "hwmeand"})
        .T
    )


def get_hwi(hw: pd.DataFrame, time_scale: str):
    """
    HWI: Total intensity of heatwaves for the time_scale period.
    Intensity is the sum of anomalies throught the heatwave event.
    """
    hw["sum_anomalies"] = hw["duration"] * hw["mean_anomaly"]
    hwi = hw.groupby(time_scale)["sum_anomalies"].sum()
    return hwi.to_frame().rename(columns={"sum_anomalies": "hwi"}).T


def get_hwmaxi(hw: pd.DataFrame, time_scale: str):
    """
    HWMAXI: Maximum intensity of heatwaves for the time_scale period.
    Intensity is the sum of anomalies throught the heatwave event.
    """
    hw["sum_anomalies"] = hw["duration"] * hw["mean_anomaly"]
    hwmaxi = hw.groupby(time_scale)["sum_anomalies"].max()
    return hwmaxi.to_frame().rename(columns={"sum_anomalies": "hwmaxi"}).T


def get_hwmeani(hw: pd.DataFrame, time_scale: str):
    """
    HWMEANI: Mean intensity of heatwaves for the time_scale period.
    Intensity is the sum of anomalies throught the heatwave event.
    """
    hw["sum_anomalies"] = hw["duration"] * hw["mean_anomaly"]
    hwmeani = hw.groupby(time_scale)["sum_anomalies"].mean()
    return hwmeani.to_frame().rename(columns={"sum_anomalies": "hwmeani"}).T


def _get_timestep(hw: pd.DataFrame, time_scale: str, date_col: str) -> pd.DataFrame:
    """Add time scale column to the dataframe."""
    if time_scale == "year":
        hw[time_scale] = hw[date_col].dt.strftime("%Y")
    elif time_scale == "month":
        hw[time_scale] = hw[date_col].dt.strftime("%Y-%m")
    elif time_scale == "decade":
        hw[time_scale] = (hw[date_col].dt.year // 10) * 10
    return hw


def _create_complete_time_index(target_years: tuple[int, int], time_scale: str) -> pd.Index:
    """
    Create a complete time index for the specified target years and time scale.

    Parameters
    ----------
    target_years : tuple[int, int]
        Range of years (start_year, end_year) inclusive.
    time_scale : str
        Time scale: "year", "month", or "decade".

    Returns
    -------
    pd.Index
        Complete time index with no gaps.
    """
    start_year, end_year = target_years

    if time_scale == "year":
        # Create yearly index: ["2000", "2001", "2002", ...]
        years = range(start_year, end_year + 1)
        return pd.Index([str(year) for year in years])

    elif time_scale == "month":
        # Create monthly index: ["2000-01", "2000-02", ..., "2001-01", ...]
        start_date = pd.Timestamp(f"{start_year}-01-01")
        end_date = pd.Timestamp(f"{end_year}-12-31")
        date_range = pd.date_range(
            start=start_date, end=end_date, freq="MS"
        )  # Month start frequency
        return pd.Index(date_range.strftime("%Y-%m"))

    elif time_scale == "decade":
        # Create decade index: [1990, 2000, 2010, ...]
        start_decade = (start_year // 10) * 10
        end_decade = (end_year // 10) * 10
        decades = range(start_decade, end_decade + 10, 10)
        return pd.Index(list(decades))

    else:
        raise ValueError(f"Unknown time_scale: {time_scale}")

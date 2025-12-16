import xarray as xr
import argparse
import os
import sys

def check_31st_days(file_path):
    """
    Checks if a NetCDF file contains data for the 31st of months with 31 days.
    """
    # Months that strictly have 31 days
    LONG_MONTHS = [1, 3, 5, 7, 8, 10, 12]
    
    try:
        # Open dataset lazily (decode_times=True is default but explicit here)
        # chunks={} ensures we use dask and don't load data into memory
        with xr.open_dataset(file_path, chunks={}, decode_times=True) as ds:
            
            if 'time' not in ds.coords:
                print(f"[SKIP] {file_path}: No 'time' coordinate found.")
                return

            # Access the time index (pandas DatetimeIndex)
            # This is fast as it only reads metadata
            time_index = ds.indexes['time']
            
            # Get unique years and months present in this specific file
            present_years = time_index.year.unique()
            present_months = time_index.month.unique()
            
            missing_entries = []

            for year in present_years:
                for month in LONG_MONTHS:
                    # Only check if this long month is actually supposed to be in the file
                    if month in present_months:
                        # Check if specific Year-Month-31 exists in the index
                        # Efficient lookup in pandas index
                        day_check = (time_index.year == year) & \
                                    (time_index.month == month) & \
                                    (time_index.day == 31)
                        
                        if not day_check.any():
                            missing_entries.append(f"{year}-{month:02d}")

            if missing_entries:
                print(f"[FAIL] {file_path}")
                print(f"       >>> Missing data on 31st for: {missing_entries}")
            else:
                # Optional: Comment this out to reduce log noise if you have many files
                print(f"[PASS] {file_path}")

    except Exception as e:
        print(f"[ERROR] Could not process {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Check NetCDF files for missing 31st days.")
    parser.add_argument("--base-path", type=str, required=True, 
                        help="Root directory to search for .nc files")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.base_path):
        print(f"Error: Directory {args.base_path} does not exist.")
        sys.exit(1)

    print(f"Starting check on directory: {args.base_path}")
    print("-" * 60)

    # Walk through the directory structure
    for root, dirs, files in os.walk(args.base_path):
        for file in files:
            if file.endswith(".nc"):
                full_path = os.path.join(root, file)
                check_31st_days(full_path)

if __name__ == "__main__":
    main()

#python3 scripts/datasets/check_era5_dates.py --base-path /onr/data/01_raw/era5/reanalysis-era5-single-levels/
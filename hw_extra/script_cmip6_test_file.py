#!/usr/bin/env python3
"""
CMIP6 Data Validation Script

This script validates CMIP6 climate model data by:
1. Checking that all directories are not empty
2. Verifying continuous year ranges in NetCDF filenames

Expected directory structure:
01_raw/CMIP6/{model}/{experiment}/{frequency}/{variable}/

Expected filename format:
variable_timefreq_model_experiment_variant_grid_YYYYMM-YYYYMM.nc
"""


import os
import re
from pathlib import Path
import argparse


def is_raw_path(base_path):
    """
    Determine if the base path is for intermediate data.
    
    Args:
        base_path (Path): Base path to check
        
    Returns:
        bool: True if intermediate, False if raw
    """
    return '01_raw' in str(base_path) or 'raw' in str(base_path).lower()


def extract_date_range(filename):
    """
    Extract start and end dates from CMIP6 filename.

    Args:
        filename (str): NetCDF filename

    Returns:
        tuple: (start_year, start_month, end_year, end_month) or None if pattern not found
    """
    # Pattern to match YYYYMM-YYYYMM format at the end of filename
    pattern = r"(\d{6})-(\d{6})\.nc$"
    match = re.search(pattern, filename)

    if match:
        start_date, end_date = match.groups()
        start_year = int(start_date[:4])
        start_month = int(start_date[4:6])
        end_year = int(end_date[:4])
        end_month = int(end_date[4:6])
        return start_year, start_month, end_year, end_month

    return None


def extract_date_range_daily(filename):
    """
    Extract start and end dates from CMIP6 filename.
    Supports both YYYYMM-YYYYMM and YYYYMMDD-YYYYMMDD formats.

    Args:
        filename (str): NetCDF filename

    Returns:
        tuple: (start_year, start_month, end_year, end_month) or None if pattern not found
    """
    # First try YYYYMMDD-YYYYMMDD format (daily data)
    pattern_daily = r"(\d{8})-(\d{8})\.nc$"
    match = re.search(pattern_daily, filename)

    if match:
        start_date, end_date = match.groups()
        start_year = int(start_date[:4])
        start_month = int(start_date[4:6])
        int(start_date[6:8])
        end_year = int(end_date[:4])
        end_month = int(end_date[4:6])
        int(end_date[6:8])
        return start_year, start_month, end_year, end_month

    return None


def extract_date_range_hourly(filename):
    """
    Extract start and end dates from CMIP6 filename.
    Supports both YYYYMM-YYYYMM and YYYYMMDD-YYYYMMDD formats.

    Args:
        filename (str): NetCDF filename

    Returns:
        tuple: (start_year, start_month, end_year, end_month) or None if pattern not found
    """
    # First try YYYYMMDD-YYYYMMDD format (daily data)
    pattern_daily = r"(\d{12})-(\d{12})\.nc$"
    match = re.search(pattern_daily, filename)

    if match:
        start_date, end_date = match.groups()
        start_year = int(start_date[:4])
        start_month = int(start_date[4:6])
        end_year = int(end_date[:4])
        end_month = int(end_date[4:6])
        return start_year, start_month, end_year, end_month

    return None


def extract_year_intermediate(filename):
    """
    Extract year from intermediate CMIP6 filename.
    
    Args:
        filename (str): NetCDF filename (format: YYYY.nc)
        
    Returns:
        int: Year or None if pattern not found
    """
    pattern = r'^(\d{4})\.nc$'
    match = re.match(pattern, filename)
    
    if match:
        return int(match.group(1))
    
    return None


def get_month_sequence(start_year, start_month, end_year, end_month):
    """
    Generate sequence of (year, month) tuples for the given range.
    
    Args:
        start_year, start_month, end_year, end_month (int): Date range
        
    Returns:
        list: List of (year, month) tuples
    """
    months = []
    year, month = start_year, start_month
    
    while (year, month) <= (end_year, end_month):
        months.append((year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1
    
    return months


def check_continuous_coverage_raw(file_ranges):
    """
    Check if file date ranges provide continuous temporal coverage for raw data.
    
    Args:
        file_ranges (list): List of (filename, start_year, start_month, end_year, end_month)
        
    Returns:
        tuple: (is_continuous, gaps, overlaps, total_coverage)
    """
    if not file_ranges:
        return False, [], [], set()
    
    # Sort files by start date
    file_ranges.sort(key=lambda x: (x[1], x[2]))
    
    # Get all months covered by each file
    all_covered_months = set()
    file_coverages = []
    
    for filename, start_year, start_month, end_year, end_month in file_ranges:
        file_months = set(get_month_sequence(start_year, start_month, end_year, end_month))
        file_coverages.append((filename, file_months))
        all_covered_months.update(file_months)
    
    # Check for overlaps
    overlaps = []
    for i, (file1, months1) in enumerate(file_coverages):
        for j, (file2, months2) in enumerate(file_coverages[i+1:], i+1):
            overlap = months1.intersection(months2)
            if overlap:
                overlaps.append((file1, file2, sorted(overlap)))
    
    # Check for gaps in continuous coverage
    if all_covered_months:
        min_year = min(year for year, month in all_covered_months)
        max_year = max(year for year, month in all_covered_months)
        min_month = min(month for year, month in all_covered_months if year == min_year)
        max_month = max(month for year, month in all_covered_months if year == max_year)
        
        expected_months = set(get_month_sequence(min_year, min_month, max_year, max_month))
        gaps = sorted(expected_months - all_covered_months)
    else:
        gaps = []
    
    is_continuous = len(gaps) == 0
    
    return is_continuous, gaps, overlaps, all_covered_months


def check_continuous_coverage_intermediate(years):
    """
    Check if years provide continuous temporal coverage for intermediate data.
    
    Args:
        years (list): List of (filename, year) tuples
        
    Returns:
        tuple: (is_continuous, gaps, year_range)
    """
    if not years:
        return False, [], set()
    
    year_set = set(year for _, year in years)
    
    if year_set:
        min_year = min(year_set)
        max_year = max(year_set)
        expected_years = set(range(min_year, max_year + 1))
        gaps = sorted(expected_years - year_set)
    else:
        gaps = []
    
    is_continuous = len(gaps) == 0
    
    return is_continuous, gaps, year_set


def validate_directory(dir_path, is_raw, verbose=False):
    """
    Validate a single directory containing NetCDF files.
    
    Args:
        dir_path (Path): Path to directory
        is_raw (bool): True if is_raw data, False if raw
        verbose (bool): Print detailed information
        
    Returns:
        dict: Validation results
    """
    results = {
        'path': str(dir_path),
        'is_empty': True,
        'file_count': 0,
        'valid_files': 0,
        'invalid_files': [],
        'is_continuous': False,
        'gaps': [],
        'overlaps': [],
        'coverage': set()
    }
    
    if not dir_path.exists():
        results['error'] = 'Directory does not exist'
        return results
    
    nc_files = list(dir_path.glob('*.nc'))
    results['file_count'] = len(nc_files)
    results['is_empty'] = len(nc_files) == 0
    
    if results['is_empty']:
        return results
    
    if not is_raw:
        # Handle intermediate data (yearly files)
        years = []
        for nc_file in nc_files:
            year = extract_year_intermediate(nc_file.name)
            if year is not None:
                years.append((nc_file.name, year))
                results['valid_files'] += 1
            else:
                results['invalid_files'].append(nc_file.name)
                if verbose:
                    print(f"  WARNING: Could not extract year from {nc_file.name}")
        
        if years:
            is_continuous, gaps, coverage = check_continuous_coverage_intermediate(years)
            results['is_continuous'] = is_continuous
            results['gaps'] = gaps
            results['coverage'] = coverage
    else:
        # Handle raw data (date range files)
        file_ranges = []
        for nc_file in nc_files:
            monthly = extract_date_range(nc_file.name)
            daily = extract_date_range_daily(nc_file.name)
            hourly = extract_date_range_hourly(nc_file.name)
            date_info = monthly or daily or hourly
            if date_info:
                file_ranges.append((nc_file.name,) + date_info)
                results['valid_files'] += 1
            else:
                results['invalid_files'].append(nc_file.name)
                if verbose:
                    print(f"  WARNING: Could not extract date from {nc_file.name}")
        
        if file_ranges:
            is_continuous, gaps, overlaps, coverage = check_continuous_coverage_raw(file_ranges)
            results['is_continuous'] = is_continuous
            results['gaps'] = gaps
            results['overlaps'] = overlaps
            results['coverage'] = coverage
    
    return results


def get_leaf_directories(base_path):
    """
    Recursively find all leaf directories (directories containing .nc files).
    
    Args:
        base_path (Path): Base path to start search
        
    Returns:
        list: List of Path objects for leaf directories
    """
    leaf_dirs = []
    
    for item in base_path.rglob('*'):
        if item.is_dir():
            # Check if this directory contains .nc files
            nc_files = list(item.glob('*.nc'))
            if nc_files:
                leaf_dirs.append(item)
    
    return leaf_dirs


def validate_cmip6_data(base_path, verbose=False):
    """
    Validate CMIP6 data structure.
    
    Args:
        base_path (str): Base path to CMIP6 data
        verbose (bool): Print detailed information
        
    Returns:
        dict: Overall validation results
    """
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"ERROR: Base path {base_path} does not exist!")
        return {}
    
    is_raw = is_raw_path(base_path)
    data_type = "RAW" if is_raw else "INTERMEDIATE/PRIMARY"
    
    print(f"Validating {data_type} CMIP6 data in: {base_path}")
    print("=" * 60)
    
    results = {
        'total_directories': 0,
        'empty_directories': [],
        'invalid_directories': [],
        'valid_directories': 0,
        'continuous_directories': 0,
        'directories_with_gaps': [],
        'directories_with_overlaps': []
    }
    
    # Get all leaf directories (directories containing .nc files)
    leaf_dirs = get_leaf_directories(base_path)
    
    for leaf_dir in leaf_dirs:
        results['total_directories'] += 1
        relative_path = leaf_dir.relative_to(base_path)
        
        if verbose:
            print(f"\nValidating: {relative_path}")
        
        dir_results = validate_directory(leaf_dir, is_raw, verbose)
        
        if dir_results['is_empty']:
            results['empty_directories'].append(str(relative_path))
            if not verbose:
                print(f"EMPTY: {relative_path}")
        elif dir_results['invalid_files']:
            results['invalid_directories'].append(str(relative_path))
            if not verbose:
                print(f"INVALID FILES: {relative_path}")
        else:
            results['valid_directories'] += 1
            
            if dir_results['is_continuous']:
                results['continuous_directories'] += 1
                if not verbose:
                    if not is_raw:
                        years = sorted(dir_results['coverage'])
                        year_range = f"{min(years)}-{max(years)}" if years else "N/A"
                        print(f"OK: {relative_path} ({dir_results['valid_files']} files, {year_range})")
                    else:
                        print(f"OK: {relative_path} ({dir_results['valid_files']} files)")
            else:
                if dir_results['gaps']:
                    results['directories_with_gaps'].append(str(relative_path))
                    if not verbose:
                        if not is_raw:
                            print(f"GAPS: {relative_path} (missing {len(dir_results['gaps'])} years)")
                        else:
                            print(f"GAPS: {relative_path} (missing {len(dir_results['gaps'])} months)")
                
                if dir_results.get('overlaps'):
                    results['directories_with_overlaps'].append(str(relative_path))
                    if not verbose:
                        print(f"OVERLAPS: {relative_path}")
        
        if verbose:
            print(f"  Files: {dir_results['file_count']} total, {dir_results['valid_files']} valid")
            if dir_results['gaps']:
                if not is_raw:
                    print(f"  Gaps: Missing years: {dir_results['gaps']}")
                else:
                    print(f"  Gaps: {len(dir_results['gaps'])} missing months")
            if dir_results.get('overlaps'):
                print(f"  Overlaps: {len(dir_results['overlaps'])} file pairs")
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"{data_type} DATA VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total directories checked: {results['total_directories']}")
    print(f"Empty directories: {len(results['empty_directories'])}")
    print(f"Directories with invalid files: {len(results['invalid_directories'])}")
    print(f"Valid directories: {results['valid_directories']}")
    print(f"Directories with continuous coverage: {results['continuous_directories']}")
    print(f"Directories with temporal gaps: {len(results['directories_with_gaps'])}")
    
    if is_raw:
        print(f"Directories with overlapping files: {len(results['directories_with_overlaps'])}")
    
    if results['empty_directories']:
        print(f"\nEmpty directories:")
        for empty_dir in results['empty_directories']:
            print(f"  - {empty_dir}")
    
    if results['directories_with_gaps']:
        print(f"\nDirectories with temporal gaps:")
        for gap_dir in results['directories_with_gaps']:
            print(f"  - {gap_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Validate CMIP6 climate data structure and temporal continuity')
    parser.add_argument('--base-path', 
                       help='Base path to CMIP6 data (e.g., 01_raw/CMIP6/ or 02_intermediate/CMIP6/)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Print detailed validation information')
    
    args = parser.parse_args()
    
    validate_cmip6_data(args.base_path, False)


if __name__ == '__main__':
    main()
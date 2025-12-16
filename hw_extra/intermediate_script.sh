#!/bin/bash

# job name
#SBATCH --job-name=cmip6-driver

# output file
#SBATCH --output=logs/%j.txt

# job queue
#SBATCH --partition=512x1024

# CPUs requests
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1

# environemnt setup
# export $(grep -v '^\s*#' etc/.env | xargs)
# source .venv/bin/activate


declare -A VAR_DRIVERS
VAR_DRIVERS["nino34"]="local_climate"
VAR_DRIVERS["nino12"]="local_climate"
VAR_DRIVERS["anom_pdo"]="local_climate"
VAR_DRIVERS["anom_dmi_east"]="local_climate"
VAR_DRIVERS["anom_dmi_west"]="local_climate"
VAR_DRIVERS["anom_wind_pressure"]="local_climate"
VAR_DRIVERS["anom_psl_sam_40"]="local_climate"
VAR_DRIVERS["anom_psl_sam_65"]="local_climate"
VAR_DRIVERS["anom_wind_cl_raco"]="local_climate"
VAR_DRIVERS["anom_wind_cl_puelche"]="local_climate"
VAR_DRIVERS["anom_ta_cl"]="local_climate"
VAR_DRIVERS["south_pacific_always_detect"]="pacific_anticyclone"
VAR_DRIVERS["default"]="advection_mean"
VAR_DRIVERS["adv_chile"]="advection_mean"

declare -A VAR_VARIABLES
VAR_VARIABLES["nino34"]="tos"
VAR_VARIABLES["nino12"]="tos"
VAR_VARIABLES["anom_pdo"]="tos"
VAR_VARIABLES["anom_dmi_east"]="tos"
VAR_VARIABLES["anom_dmi_west"]="tos"
VAR_VARIABLES["anom_wind_pressure"]="psl"
VAR_VARIABLES["anom_psl_sam_40"]="psl"
VAR_VARIABLES["anom_psl_sam_65"]="psl"
VAR_VARIABLES["anom_wind_cl_raco"]="ua"
VAR_VARIABLES["anom_wind_cl_puelche"]="ua"
VAR_VARIABLES["anom_ta_cl"]="ta"
VAR_VARIABLES["south_pacific_always_detect"]="psl"
VAR_VARIABLES["default"]="ua,ta"
VAR_VARIABLES["adv_chile"]="ua,ta"

# Define 'calculate_kwargs' for each variant
declare -A VAR_CALCULATE_KWARGS
VAR_CALCULATE_KWARGS["nino34"]="resample_freq=1MS;rolling_window=1;lon_box=(190,240);lat_box=(-5,5);anomalies=True"
VAR_CALCULATE_KWARGS["nino12"]="resample_freq=1MS;rolling_window=1;lon_box=(270,280);lat_box=(-10,0);anomalies=True"
VAR_CALCULATE_KWARGS["anom_pdo"]="resample_freq=1MS;rolling_window=1;lon_box=(120,240);lat_box=(20,60);anomalies=True"
VAR_CALCULATE_KWARGS["anom_dmi_east"]="resample_freq=1MS;rolling_window=1;lon_box=(90,110);lat_box=(-10,0);anomalies=True"
VAR_CALCULATE_KWARGS["anom_dmi_west"]="resample_freq=1MS;rolling_window=1;lon_box=(50,70);lat_box=(-10,10);anomalies=True"
VAR_CALCULATE_KWARGS["anom_wind_pressure"]="resample_freq=1MS;rolling_window=1;lon_box=(286,292);lat_box=(-52,-40);anomalies=True"
VAR_CALCULATE_KWARGS["anom_psl_sam_40"]="resample_freq=1MS;rolling_window=1;lon_box=(0,360);lat_box=(-41,-39);anomalies=True"
VAR_CALCULATE_KWARGS["anom_psl_sam_65"]="resample_freq=1MS;rolling_window=1;lon_box=(0,360);lat_box=(-66,-64);anomalies=True"
VAR_CALCULATE_KWARGS["anom_wind_cl_raco"]="resample_freq=1MS;rolling_window=1;lon_box=(287.5,289.5);lat_box=(-37,-33);anomalies=True"
VAR_CALCULATE_KWARGS["anom_wind_cl_puelche"]="resample_freq=1MS;rolling_window=1;lon_box=(286.5,288.5);lat_box=(-42,-37);anomalies=True"
VAR_CALCULATE_KWARGS["anom_ta_cl"]="resample_freq=1MS;rolling_window=1;lon_box=(286,288);lat_box=(-42,-33);anomalies=True"
VAR_CALCULATE_KWARGS["south_pacific_always_detect"]="always_detect=True"
VAR_CALCULATE_KWARGS["default"]=""
VAR_CALCULATE_KWARGS["adv_chile"]="lon_box=(282,286);lat_box=(-42,-33)"


# Static parameters (don't change per variant)
start_date="2015-01-01"
end_date="2100-12-31"
models=("INM-CM5-0")
scenarios=("ssp126" "ssp245" "ssp370" "ssp585")

# Load parameters from the associative arrays based on RUN_VARIANT
variant="adv_chile"
driver="${VAR_DRIVERS[$variant]}"
variables="${VAR_VARIABLES[$variant]}"
calculate_kwargs="${VAR_CALCULATE_KWARGS[$variant]}"
load_kwargs="use_historical=True;historical_start_date=1980-01-01;historical_end_date=2014-01-01"
# driver="high_low_difference"
# variables="psl"
# calculate_kwargs=""

# --- Safety Check ---
# Check if the variant definition exists
if [[ -z "$variables" ]]; then
    echo "Error: Parameters for variant '$variant' are not defined."
    echo "Please define them in the 'VAR_...' arrays at the top of the script."
    exit 1
fi

# --- Run the Transformation ---
echo "--- Starting job for variant: $variant ---"
for model in "${models[@]}"; do
    for scenario in "${scenarios[@]}"; do
        optional_args=()
        
        # 2. Check if variables string contains "ua" OR "ta"
        if [[ "$variables" == *"ua"* || "$variables" == *"ta"* ]]; then
            
            # 3. If it does, determine pressure_level based on the model
            pressure_level=""
            case "$model" in
                "ACCESS-ESM1-5")
                    pressure_level="85000.00000001001"
                    ;;
                "IPSL-CM6A-LR" | "MPI-ESM1-2-LR" | "INM-CM5-0")
                    pressure_level="85000.0"
                    ;;
                *)
                    # Default for all other models
                    pressure_level="85000"
                    ;;
            esac
            
            # 4. Add the pressure level argument to our optional array
            optional_args+=(--pressure_level "$pressure_level")
            echo "Processing $model with pressure level: $pressure_level"
        else
            echo "Processing $model (no pressure level required)"
        fi

        python3 scripts/transformations/cmip6_intermediate_primary.py \
            --driver "$driver" \
            --start_date "$start_date" \
            --end_date "$end_date" \
            --model "$model" \
            --scenario "$scenario" \
            --variables "$variables" \
            --variant "$variant" \
            --calculate_kwargs "$calculate_kwargs" \
            --load_kwargs "$load_kwargs" \
            "${optional_args[@]}" # Safely adds pressure_level args only if set
    done
done

echo "--- Job finished ---"
# ACCESS-ESM1-5 85000.00000001001
# IPSL-CM6A-LR" "MPI-ESM1-2-LR 85000.0

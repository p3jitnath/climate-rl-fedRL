#!/bin/bash

BASE_DIR="/gws/nopw/j04/ai4er/users/pn341/climate-rl-fedrl"
ALGOS=("ddpg" "td3" "tqc")

GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[1;33m"
NC="\033[0m"

declare -a missing_step_files
missing_step_files=()

# Print header
printf "%-60s" "Run Directory"
for algo in "${ALGOS[@]}"; do
    printf "%15s" "$algo"
done
echo

# Loop over top-level run directories
for top_dir in "$BASE_DIR"/records/*ebm-v2*/; do
    run_name=$(basename "$top_dir")
    printf "%-60s" "$run_name"
    failed_any_algo=false

    for algo in "${ALGOS[@]}"; do
        # Decide step file name
        if [[ "$run_name" == *infx* ]]; then
            step_file="step_200.pth"
        else
            step_file="step_20000.pth"
        fi

        # Find all matching run dirs for the algo
        run_dirs=$(find "$top_dir" -mindepth 1 -maxdepth 1 -type d -name "*_${algo}_*")
        step_file_count=0

        for run_dir in $run_dirs; do
            step_path="$run_dir/$step_file"
            if [[ -f "$step_path" ]]; then
                ((step_file_count++))
            else
                missing_step_files+=("$step_path")
            fi
        done

        if [[ ( "$step_file_count" -eq 20 && "$run_name" == *a2* ) || ( "$step_file_count" -eq 60 && "$run_name" == *a6* ) ]]; then
            printf "${GREEN}%15s${NC}" "$step_file_count"
        else
            printf "${RED}%15d${NC}" "$step_file_count"
            failed_any_algo=true
        fi
    done
    echo
done

# Print missing step files with last 3 path components
echo -e "${YELLOW}Missing:${NC}"
if [ ${#missing_step_files[@]} -eq 0 ]; then
    echo "-"
else
    for path in "${missing_step_files[@]}"; do
        echo "$path" | awk -F/ '{n=NF; print $(n-3)"/"$(n-2)"/"$(n-1)}'
    done
fi

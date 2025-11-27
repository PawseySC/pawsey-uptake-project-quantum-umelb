#/bin/bash -l

hname=$(hostname)
if [[ "$hname" = "setonix-07" ||  "$hname" = "setonix-08" ]]; then 
    echo "Starting TRAF workflow on ${hname}"
    . ~/.bashrc
    load_modules_for prefect2
    module load singularity/4.1.0-nompi 
    ../../qbitbridge/workflow/scripts/run_start_up_scripts_for_setonix.sh
    sleep 2
    export PREFECT_API_URL=http://${hname}:4200/api
    python3 traf_flow.py --output_dir /scratch/bq2/pelahi/traf_runs/
    ../../qbitbridge/workflow/scripts/close_and_cleanup.sh
    echo "Completed TRAF workflow on ${hname}"
else
    echo "Not on allowed system ${hname}"
fi
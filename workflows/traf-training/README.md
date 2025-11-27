# Traf workflow
This is a prefect2 orchestrated workflow for creating lots of traf runs.

## Installation

It can installed using a virtual environment. See qbitbridge for some details

## Running

It can be run as simply as
```bash
python3 traf_flow.py
```
Though it is likely useful to spin up a local prefect database and prefect server (see scripts in qbitbridge).

Make sure to copy the cluster configuration file to the qbitbridge/workflow/clusters/ directory

You can also run 
```bash
./run_traf_workflow.sh
```

This script is hard-coded somewhat to use specific scripts to start up process on setonix workflow nodes and also uses 
a bash function to load a prefect2 environment. This you might have to setup. The specific modules loaded are 

```bash
module load cmake/3.30.5
module load python/3.11.6
source $MYSOFTWARE/py-prefect2/bin/activate
export PREFECT_HOME=$MYSCRATCH/prefect2/
module load singularity/4.1.0-nompi
```
Thus requires that a virtual python environment has been setup with prefect2 installed. 


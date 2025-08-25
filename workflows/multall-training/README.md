# Multall workflow
This is a prefect2 orchestrated workflow for creating lots of multall runs.

## Installation

It can installed using a virtual environment. See qbitbridge for some details

## Running

It can be run as simply as
```bash
python3 multall_flow.py
```
Though it is likely useful to spin up a local prefect database and prefect server (see scripts in qbitbridge).

Make sure to copy the cluster configuration file to the qbitbridge/workflow/clusters/ directory

## Issues
The issue currently is that environment variables must be created to pass long file names and arguments which means you cannot concurrently 
run many instances of multall as concurrent tasks in a flow.

Possibility is to alter this to run many flows each with a single core task in a different slurm session so that the environments are isolated. Messy but might be necessary. 

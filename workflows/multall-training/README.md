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

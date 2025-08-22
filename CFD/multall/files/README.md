# Example files

This directory contains example input and output files from multall,
and scripts to manipulate them.

## Multall files

`meangen.in` is the input file to `menagen`, provides a quick starting point
for blade design.

`stagen.dat` is the input file to `stagen`, provides detailed control for geometry
definition for `multall`.

## Scripts

`read_efficiency.sh` reads the output of `multall` and reports the
"TOTAL TO TOTAL ISENTROPIC EFFICIENCY".\\
The filename is the first argument, typically `multall.txt`.

`change_inflow_angle.sh` changes the inlet flow yaw angle in the input file to
`stagen`.\\
The filename is the first argument, typically `stagen.dat`.\\
The desired yaw angle is the second argument.\\
Note: it may be useful for convergence if we manually adjust the tangential
velocity as it provides an initial guess.


## Updated interface 

New interface for all code allow environment variables to be set to specific input and output

For example, now `meangen` can take any filename (<1024 characters long) as input and can produce output that has any filename
```bash
export MEANGEN_ARGS="<inputfile> <outputfile> "
../bin/meangen > meangen.txt
```

The same inteface applies to `stagen` and `multall`
```bash
export STAGEN_ARGS="intpufile outbasefilename "
../bin/stagen > stagen.txt
export MULTALL_ARGS="intpufile outbasefilename "
../bin/multall > multall.txt
```

This allows a simple python like workflow of 
```python
job_id : int = 1
os.environ["MEANGEN_ARGS"]=f"meangen_{job_id}.in meangen_{job_id} "
os.environ["STAGEN_ARGS"]=f"meangen_{job_id}_stagen.dat stagen_out_{job_id} "
os.environ["MULTALL_ARGS"]=f"stagen_out_{job_id}_new.dat multall_{job_id} "

cmds : List[str] = ["meangen", "stagen", "multall"]
for cmd in cmds:
    process = subprocess.run(
        cmd, capture_output=add_output_to_log, text=add_output_to_log
    )
    # do some post processing of output if required
    process.stdout
```

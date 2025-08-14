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


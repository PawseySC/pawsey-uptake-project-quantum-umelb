#!/bin/bash

echo F | meangen | tee meangen.txt
echo Y | stagen | tee stagen.txt

echo N > intype # this is required for multall

# the following command activates restart capability
# useful if running consecutive similar simulations
# requires a previous `flow_out` file
#sed -E -i "/IF_RESTART/!b;N;s/0/$1/;P;D" stage_new.dat

# note that interface has changed
# multall < stage_new.dat | tee multall.txt

multall 

# note that key summary data is located in `multall.txt`, that is produced to stdout
# this file will contain relevant information
# this is the file we will process. 

# the file can be captured using read_efficiency.sh, with the idea of capturing the last output
# of the efficiency. 



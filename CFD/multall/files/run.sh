#!/bin/bash

echo F | meangen | tee meangen.txt
echo Y | stagen | tee stagen.txt

echo N > intype # this is required for multall

# the following command activates restart capability
# useful if running consecutive similar simulations
# requires a previous `flow_out` file
#sed -E -i "/IF_RESTART/!b;N;s/0/$1/;P;D" stage_new.dat

multall < stage_new.dat | tee multall.txt


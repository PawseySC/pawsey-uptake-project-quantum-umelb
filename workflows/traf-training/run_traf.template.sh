#!/bin/bash

#set a path
export PATH_EXE=/software/projects/bq2/mrosenzweig/uptake/executables
# got to local directory
SCRIPT_DIR=$(dirname "$0")
cd $SCRIPT_DIR

#run executables from output dir 
# blade_design tools is what is supposed to create the airfoil.dat file
# this needs to be pulled in to process the airfoil_config.yaml 

$PATH_EXE/jerryo210310.x
$PATH_EXE/tomo.x < datatomo
$PATH_EXE/addrad.x < dataadd
cp -s meshq3d fort.12
$PATH_EXE/stitcho-q3d.x < datastitch_inlet
cp ./mq3d_stitch.dat fort.11
$PATH_EXE/stitcho-q3d.x < datastitch_outlet
cp ./mq3d_stitch.dat fort.13
cp -s xy.dat xy.xyz
$PATH_EXE/trafq3d22_14122022/trafq3d22.x 
cp -s xy.dat xy.xyz
cp -s fort.40 fort.50
cp -s fort.40 fort.70
$PATH_EXE/pstg2d14a_moverows.x < datapstg
cp -s fort.62 flow.dat
$PATH_EXE/outq3d12c.x
$PATH_EXE/pstg2d14a_moverows.x < datapstg


#!/bin/bash

# changes inlet yaw angle (tangential / meridinal)

if [[ $# -ne 2 ]]; then
  echo "Usage: ./change_inflow_angle.sh <filename> <yaw angle>"
  echo "Error: 2 arguments expected"
  exit 1
fi

num='([+-]?[0-9]+\.?[0-9]*[eE]?[+-]?[0-9]*)'

file=$1
sed -Ei "/YAW ANGLE IN/s/$num/$2/g" $file


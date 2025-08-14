#!/bin/bash

# reads total to total isentropic efficiency

if [[ $# -ne 1 ]]; then
  echo "Usage: ./read_efficiency.sh <filename>"
  echo "Error: One argument expected"
  exit 1
fi

num='([+-]?[0-9]+\.?[0-9]*[eE]?[+-]?[0-9]*)'

file=$1
rline=$(sed -En "/RESULTS FOR STAGE NUMBER\s+1/=" $file | tail -1)
nis=$(sed -En "$rline,\$s/TOTAL TO TOTAL ISENTROPIC EFFICIENCY[ =]*$num.*/\1/p" $file)
echo $nis


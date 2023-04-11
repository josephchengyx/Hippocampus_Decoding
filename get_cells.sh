#!/bin/bash
day=$1
curr=$PWD
dest="/Volumes/Hippocampus/Data/picasso-misc/$day/session01"
cd $dest
find . -type d -name "cell0*" | cut -d "/" -f 2-4 > ~/Documents/neural_decoding/scripts/cell_list.txt
cd $curr

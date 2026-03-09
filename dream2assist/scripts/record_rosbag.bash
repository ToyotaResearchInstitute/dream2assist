#!/bin/bash

# Check whether a log name was provided, otherwise default to "temp"
PID="temp"
if [ ! -z "$1" ]
    then
        PID="$1"
fi

timestamp=$(date +%Y%d%m_%H%M%S)

ros2 bag record -a --output $HOME/sdm_demo/rosbags/$timestamp\_$PID -s mcap -d 300

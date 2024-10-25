#!/bin/bash
f="$HOME/sim_ws/src/f1tenth_gym_ros/config/sim.yaml"

if [ "$1" = "-p" ]; then
	grep ".*_map" $f
	exit 0
fi

cp $f $f.copy

sed -i -e "s/maps\/.*_map/maps\/${1}_map/g" $f

cd ~/sim_ws 

colcon build --packages-select f1tenth_gym_ros
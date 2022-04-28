#!/bin/bash
while true;
do
		if [ -z "$(pgrep -u amacke26 python)" ];
		then
				echo "Training crashed..."
				python anime_training.py 
		fi
		sleep 10;
done

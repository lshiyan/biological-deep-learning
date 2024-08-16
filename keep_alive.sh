#!/bin/bash

# Set the interval (in seconds) between each clear command
INTERVAL=300  # 300 seconds = 5 minutes

# Infinite loop
while true; do
    clear  # Sends the clear command to the terminal
    sleep $INTERVAL  # Waits for the specified interval
done
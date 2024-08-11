#!/bin/bash

while true; do
    python test_npy_alignment.py
    # Check the exit code of the Python script
    if [ $? -eq 0 ]; then
        break
    fi
done
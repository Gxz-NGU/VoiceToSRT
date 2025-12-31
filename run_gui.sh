#!/bin/bash
# Script to run VoiceToSRT GUI with the correct environment

# Path to the python executable that has dependencies installed
PYTHON_EXEC="/opt/miniconda3/bin/python3"

# Check if python executable exists
if [ ! -f "$PYTHON_EXEC" ]; then
    echo "Error: Python executable not found at $PYTHON_EXEC"
    echo "Please ensure Miniconda is installed or update the path in this script."
    exit 1
fi

# Run the GUI
echo "Starting VoiceToSRT GUI..."
$PYTHON_EXEC gui.py

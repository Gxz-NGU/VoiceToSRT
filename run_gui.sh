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

# Work around OpenMP duplicate runtime crash on macOS.
export KMP_DUPLICATE_LIB_OK=TRUE
export KMP_INIT_AT_FORK=FALSE
export OMP_NUM_THREADS=1
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_THREAD_LIMIT=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Run the GUI
echo "Starting VoiceToSRT GUI..."
$PYTHON_EXEC gui.py

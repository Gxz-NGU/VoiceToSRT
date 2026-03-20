#!/bin/bash
# Script to run VoiceToSRT GUI with the correct environment

if [ -n "$PYTHON_EXEC" ] && [ -x "$PYTHON_EXEC" ]; then
    :
elif [ -x "/opt/miniconda3/bin/python3" ]; then
    PYTHON_EXEC="/opt/miniconda3/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_EXEC="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON_EXEC="$(command -v python)"
else
    echo "Error: Python executable not found."
    echo "Set PYTHON_EXEC or activate an environment that provides python3."
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
"$PYTHON_EXEC" gui.py

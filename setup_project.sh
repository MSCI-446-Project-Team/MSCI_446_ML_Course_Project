#!/bin/bash

# Check if Python is installed
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null
then
    echo "Python is not installed. Installing Python..."
    # Add Python installation command here (depends on the OS)
else
    echo "Python is already installed."
fi

# Check if pip is installed
if ! command -v pip &> /dev/null
then
    echo "pip is not installed. Installing pip..."
    # Python 2.x
    if command -v python &> /dev/null; then
        curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        python get-pip.py
    # Python 3.x
    elif command -v python3 &> /dev/null; then
        curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        python3 get-pip.py
    fi
    rm get-pip.py
else
    echo "pip is already installed."
fi

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    if command -v python3 &> /dev/null; then
        python3 -m venv venv
    else
        python -m venv venv
    fi
fi

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "requirements.txt not found."
fi



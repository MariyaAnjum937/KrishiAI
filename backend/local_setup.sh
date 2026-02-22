#!/bin/sh

echo "===============================" 
echo "welcome to local setup" 
echo "This sets up the local environment and installs the required libraries" 
echo "-------------------------------" 

if [ -d '.venv' ]
then 
    echo "virtual environment already exists" 
else 
    echo "no virtual environment found, creating one" 
    python3 -m venv .venv 
fi 

echo "Activating the virtual environment" 
. .venv/bin/activate

echo "installing the required libraries" 
pip install --upgrade pip 
pip install -r requirements.txt 

pip install --upgrade pip --break-system-packages
pip install -r requirements.txt --break-system-packages


echo "Setup Complete" 
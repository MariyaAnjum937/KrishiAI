#!/bin/sh

echo "====================================" 
echo "Running your application locally" 
echo "-----------------------------------" 

if [ -d '.venv' ]
then 
    echo "local environment found" 
else 
    echo "no local environment found, please run local_setup.sh first" 
    exit 1
fi 

echo "Activating the local environment" 
. .venv/bin/activate 

export MODEL_PATH=mobilenetv2_best.keras

echo "running your application" 
python3 main.py 
#!/bin/bash
####################################################################
# Author:       Shane Miller
# Course:       COMP6830 Data Science Capstone Project II
# Purpose:      To auto-install program dependencies and to launch 
#               application.
# Description:  Program checks to see if virtual environment 
#               folder is present, activates it then installs 
#               program dependencies and launches application.
####################################################################

pVer=$(python3 -V 2>&1 | sed 's/.* \([0-9]\).\([0-9]\).*/\1\2/')
psVer=${pVer: -1}

if [ -d "${PWD}/venv" ]
then
    source venv/bin/activate
    python3 wsgi.py
else
    # Install venv for your python version
    sudo apt-get install python3.${psVer}-venv
    
    python3 -m venv venv
    source venv/bin/activate
    python3 -m pip install --upgrade pip
    
    # Program dependencies are in txt file
    pip install -r requirements.txt
    
    if [ ! -f ".env" ]
    then
        # Create environment file
        touch ".env"                            
        echo 'FLASK_ENV=production' >> ".env"
        echo 'SECRET_KEY=YOURSECRETKEY' >> ".env"
    fi
    python3 wsgi.py
fi
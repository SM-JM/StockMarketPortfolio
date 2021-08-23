########################################################################
# Author:       Shane Miller
# Course:       COMP6830 Data Science Capstone Project II
# Purpose:      To create flask application object and launch application
# Description:  Creates flask instance and runs it on the IP address 
#               assigned to the current machine (or localhost if the
#               are running it from the same machine), using post 5000.
#               
#               folder is present, activates it then installs 
#               program dependencies and launches application.
########################################################################

# App entry point."""
from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000,debug=True)
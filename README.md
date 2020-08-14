# greyhounds
Capstone project for the Udacity Machine Learning Engineer Nano Degree

# Clone repo and create directories

`mkdir repos`

`cd repos`

`git clone https://github.com/tonyjward/dogs.git`

`cd dogs`

`mkdir data`

`mkdir logs`

`mkdir dogs/tests`


# Create a virtual environment 

`cd`

`mkdir .venv`

`python3.6 -m venv .venvs/dogs`

Activate the virtual environment

`source .venv/dogs/bin/activate`

Install dependencies 

`cd repos/dogs/`

`pip3 install -r requirements.txt`

Install `dogs` package built using this project in develop model

`pip3 install -e ./`

# Create Jupyter Kernel 
`ipython kernel install --user --name=dogs`

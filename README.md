# supsys-lakhina-entropy-detection
Academic project for the course "Supervision des Systèmes et Audit de Sécurité" at IMT Atlantique, focused on implementing and evaluating a Lakhina entropy-based intrusion detection approach for botnet detection using the CTU-13 dataset.

## Setup Linux

Clone the repository:

> git clone <repo>

enter the directory:

> cd supsys-lakhina-entropy-detection

Run the setup:

> make setup

This will:
- create a virtual environment
- install all dependencies


## Run the project
> make run


## Download dataset

To download the full CTU-13 dataset:

> make download-dataset

To download a specific scenario (replace <SCENARIO> with the desired number):

> make download-scenario SCENARIO=<SCENARIO>


## Setup Windows (without Make)

Clone the repository:

> git clone <repo>

enter the directory:

> cd supsys-lakhina-entropy-detection

Create a virtual environment:

> python -m venv venv

Activate the virtual environment:

> venv\Scripts\activate

Upgrade pip:

> python -m pip install --upgrade pip

Install dependencies:

> pip install -r requirements.txt


## Run the project

Activate the virtual environment:

> venv\Scripts\activate

Run the project:

> python src/main.py


## Download dataset 
To download the full CTU-13 dataset run:

> mkdir data  
> cd data  
> curl -O https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/CTU-13-Dataset.tar.bz2  
> tar -xjf CTU-13-Dataset.tar.bz2  
> cd ..

To download a specific scenario replace <SCENARIO> with the desired number:

> mkdir data  
> cd data  
> curl -O https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-<SCENARIO>.tar.bz2  
> tar -xjf CTU-Malware-Capture-Botnet-<SCENARIO>.tar.bz2  
> cd ..


## Clean environment

To remove the virtual environment folder:

> Remove-Item -Recurse -Force venv
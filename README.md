# GymAIner
### Gym AI Trainer for Artificial Intelligence Applications Course

## Requirements
  - Python 3.7.6
  - 32 to 64 GB RAM required to properly run the training and testing processes.
	- 32 GB RAM is the minimum requirement for the project.
	- RAM with less than 32 GB could result with "RESOURCE_EXHAUSTED" exception during dataset loading, model training and testing phases.
  - Cuda supported NVIDIA GPU if possible for faster training and testing processes.
    - Doing this process on CPU instead will result only in longer time.


# Installation Process
## Installation via Visual Studio 2022
  - If you have Visual Studio 2022 installed, you can;
	- Automatically create virtual environment,
	- Install packages and,
	- Run the application from the main.py file in the project folder.

  - If not, please follow the steps in "Dataset Installation" sections below.


## Dataset Installation
  ### Dataset Link: https://drive.google.com/drive/folders/1WE0JB1N0teZHPjep_ibvpQuOEjlijeUs?usp=sharing
  - Please create a folder named 'dataset' in the project directory.
  - Then, download the 'Fit3D Video Dataset.rar' file from the link below and move it into this folder.
  - Lastly, extract the file to the current location..")


## Project Installation
  - Clone the project to your local machine by running the following command:
	```bash
       git clone https://github.com/Go4Real34/gymainer.git
	```

## Required Libraries
  - Please install the required packages by following the steps below:
	- Navigae to the project directory where 'main.py' is at.
	```bash
       cd GymAIner
	```
	
    - Create a virtual environment:
	```bash
       py -3.7 -m venv venv
	```

	- Activate the virtual environment:
	  - For Windows;
		```bash
           .\venv\Scripts\activate
		```
	  - For MacOS and Linux;
		```bash
           source venv/bin/activate
		```

	- Install the required packages:
		```bash
           pip install -r requirements.txt
		```


## Run Application
  - Run the following command after package installations are completed:
	```bash
       python main.py
	```
  - After running, navigate through the console for desired operations.


# Project Details
## Dataset
  - Dataset: Fit3D Video Dataset
  - Number of Exercises (Classes): 47
  - Number of Videos each Class Containing: 35
  - Number of Total Videos: 1645

## Model
  - Model Type: LSTM
  - Model Video Sequence Count: 75 Frames per Video
  - Trained Model Video Resolution: 128 x 128

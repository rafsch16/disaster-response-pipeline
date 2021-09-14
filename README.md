# Disaster Response Pipeline Project

### Table of Contents 
1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Instructions](#instructions)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation<a name="installation"></a>

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python. The code should run with no issues using Python versions 3.*.

## Project Motivation<a name="motivation"></a>

This project is part of the Udacity course in Data Science. The files related to this project were provided by Udacity, with modifications added by the student. The aim of the project is to analyse disaster data from Figure Eight to build a model for an API that classifies disaster data.

The model will be accesible through a web app, where an emergency worker can input a new message and get classification results in several categories. This web app will also display visualizations of the data.

## File Descriptions<a name="files"></a>

The file '../data/process_data.py' contains the ETL pipeline for the messages and categories datasets.

The file '../models/train_classifier.py' contains the machine learning pipeline to train the classifier.

The file '../app/run.py' uses Flask to run a web app.

## Instructions<a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

The disaster dataset to train the model was provided by Figure Eight. The code for the web app was provided by Udacity, where only the data visualization was modified.
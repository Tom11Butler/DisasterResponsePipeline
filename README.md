# Disaster Response Pipeline Project

## Motivation
This project takes a data set containing real messages that were sent during disaster events. A machine learning pipeline is created to categorise the events so that they can be sent to an appropriate disaster agency.

## Packages
-   Anaconda Distribution of Python
-   Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
-   Natural Language Process Libraries: NLTK
-   SQLlite Database Libraries: SQLalchemy
-   Web App and Data Visualization: Flask, Plotly

## Instructions
1. Run the following commands in the project's root directory to set up your database and model.

- To run ETL pipeline that cleans data and stores in database
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
- To run ML pipeline that trains classifier and saves
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
`python run.py`

3. Go to http://0.0.0.0:3001/

## File Structure
The overall structure of the repository is as follows:
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
```
### Python scripts
The first part of the data pipeline is the Extract, Transform, and Load process. Here, I read the dataset, clean the data, and then store it in a SQLite database. The data cleaning is done with Pandas.

For the machine learning portion, the data was split into a training set and a test set. Then, I created a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the `message` column to predict classifications for 36 categories (multi-output classification). Finally, the model was exported to a pickle file.

The file `run.py` contains the backend code for the web app. Brief data wrangling code is included to prepare data for the visualisations on the web app home page.

## Licensing
All packages are open-source. All code has been written by myself and may be used.

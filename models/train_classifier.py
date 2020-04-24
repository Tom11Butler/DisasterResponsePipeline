'''
TRAIN CLASSIFIER
Disaster Response Project
Author: Thomas Butler

Arguments:
    1) SQLite database path for pre-processed data
    2) pickle file name to save ML model

'''

import sys
import sqlalchemy
import numpy as np
import pandas as pd
import pickle
from sqlalchemy import create_engine
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report, f1_score, precision_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
nltk.download(['punkt','stopwords','wordnet','averaged_perceptron_tagger'])

def load_data(database_filepath):
    '''
    Loads the data
    
    Args:
    - database_filepath: the string filepath of where the database is
    Outputs:
    - X: feature DataFrame
    - Y: target variables DataFrame
    - category_names: all 36 category names list
    
    Functionality:
    
    '''
    # load the data from the database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    
    return X, Y, category_names

def tokenize(text):
    '''
    Tokenize function
    
    Args:
    - text: string of the text we want to tokenise, English
    Outputs:
    - list of cleaned tokens from the text
    
    Functionality:
    Identifies the regex for urls and finds all occurrences in the text
    Loops through all detected urls and replaces with 'urlplaceholder'
    Tokenises the text by words, creating a list
    Lemmatises each word, set to lower case and remove white space
    '''
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        # replace urls with a placeholder so we know if link is in message
        text = text.replace(url, "urlplaceholder")
        
    # separates text into a list of words
    tokens = word_tokenize(text)
    # reduces words to base version
    lemmatizer = WordNetLemmatizer()

    # lemmatise words, set to lower case, strip of white space
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''
    Derived estimator class for extracting the starting verb of a sentence
    
    Derived from BaseEstimator and TransformerMixin classes
    '''

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    

def build_model():
    '''
    Build model function
    
    Args:
    - none
    Outputs:
    - Scikit ML Pipeline to process text messages and appl classifier
    
    Functionality:
    
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))

    ])
    
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_estimators': [10, 50]
    }

    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates the model and outputs overall accuracy score
    
    Args:
    - model: GridSearchCV object for ML model
    - X_test: test features DataFrame
    - Y_test: target features DataFrame
    - category_names: list of all 36 category names
    Outputs:
    - none
    
    Functionality:
    Uses the model to predict the categories based off 
    '''
    
    Y_pred = model.predict(X_test)
    
    # put the data into a useful DataFrame
    Y_pred = pd.DataFrame(Y_pred, columns=category_names)
    
    # reindex Y_test so labelled identically to Y_test
    # this is needed for the direct comparison later
    Y_test = Y_test.reset_index().drop('index', axis=1)
    
    # Loop through each category and print classification matrix
    for col in category_names:
        print('Category: {}'.format(col))
        print(classification_report(y_true=Y_test[col], y_pred=Y_pred[col]))
        print('-------------------------------------------------------')
    
    # print out the overall model accuracy 
    accuracy = (Y_pred == Y_test).mean().mean()
    print("Overall Accuracy:", accuracy)
    print("\nBest Parameters:", model.best_params_)
    
    pass


def save_model(model, model_filepath):
    '''
    Save model function
    
    Args:
    - model: model Pipeline object or GridSearchCV object 
    - model_filepath: string of model save location
    Outputs:
    - none
    
    Functionality:
    Saves the model to a pickle file with name of the filepath given
    '''
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    pass


def main():
    '''
    Function that runs when the file is called to run directly
    
    Args:
    - none
    Outputs:
    - none
    
    Functionality:
    Takes arguments sent to the file for database and model filepaths
    Calls function to load the data
    Splits the data into 
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
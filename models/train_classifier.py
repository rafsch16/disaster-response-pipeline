# import libraries
import sys
import os
import pandas as pd
import re
import numpy as np
import pickle
import nltk

# download necessary NLTK data
nltk.download(['punkt','wordnet'])

# import statements SQL
from sqlalchemy import create_engine

# import NLTK statements
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# import statements sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    '''
    Loads the data from a SQL table.
    
    Parameters:
        database_filepath (str): file path to the database
    
    Returns:
        X (np.array): Explanatory variable
        Y (np.array): Response variables
        category_names (np.array): Response categories
    '''
    # obtain path to the database
    path ="sqlite:////home/workspace/"+ database_filepath
    
    # create a database engine
    engine = create_engine(path)
    
    # read the SQL table into DataFrame objects
    df = pd.read_sql_table('disaster_response_data', engine)
    X = df['message'].values
    Y = df.iloc[:,4:].values
    
    # obtain category names
    category_names = df.iloc[:,4:].columns.values
    
    return X, Y, category_names

def tokenize(text):
    '''
    Tokenize a text into words and reduce words to their root forms.
    
    Parameters:
        text (str)
    
    Returns:
        clean_tokens (list)
    '''
    # split text into words
    tokens = word_tokenize(text)
    
    # initialize lemmatizer object
    lemmatizer = WordNetLemmatizer()
    
    # lemmatize and clean words
    clean_tokens=[]
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    '''
    Builds a multi-output model for text calssification using the sklearn 
    Pipline class.
    
    The pipline includes:
    Transforms - CountVectorizer, TfidfTransformer
    Estimator - RandomForestClassifier
    
    GridSearchCV is used to find the best parameters for the model
    
    Returns:
        model (estimator object)
    '''
    # create Pipline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # specify parameter values for the grid search
    parameters = {
        'vect__max_features': (None, 5000, 10000),
        'vect__max_df': (0.5, 0.75, 1),
        'clf__estimator__min_samples_split': [10, 20, 30],
    }
    
    # create model
    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)

    return model
    

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates the f1 score, precision and recall for each category.
    
    Parameters:
        model (estimator object)
        X_test (np.array): Explanatory variable 
        Y_test (np.array): Response variables
        category_names (np.array): Category names
    '''
    # predict on test data
    Y_pred = model.predict(X_test)
    
    # print the scores for each category
    for i in range(0,Y_test.shape[1],1):
        print(category_names[i])
        print(classification_report(Y_test[:,i], Y_pred[:,i]))


def save_model(model, model_filepath):
    '''
    Saves the model.
    
    Parameters:
        model (estimator object)
        model_filepath (str): File path to save the model
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
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
import sys
import os
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Extracts data from two csv files and returns a merged dataframe 
    object 
    
    Parameters:
        messages_filepath (str): file path to the messages dataset
        categories_filepath (str): file path to the categories dataset
    
    Returns:
        df (pandas.DataFrame)
    
    '''
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = pd.merge(messages,categories,how='inner',on='id')
    
    # return dataframe
    return df


def clean_data(df):
    '''
    Transforms the data using dummy variables for the categories. 
    Removes duplicate data.
     
    Parameters:
        df (pandas.DataFrame)
    
    Returns:
        df (pandas.DataFrame)
    
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    
    # extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x[:-2]).values.tolist()
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
    # drop the original categories column from `df`
    df.drop(columns='categories',inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    '''
    Load the data into an SQLite database,
     
    Parameters:
        df (pandas.DataFrame)
        database_filename (str): file name of the database
    
    '''
    # obtain path
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path ="sqlite:///"+ os.path.join(dir_path, database_filename)
    
    # creat a database engine
    engine = create_engine(path)
    
    # output data into SQL table
    df.to_sql('disaster_response_data', engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
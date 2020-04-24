import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Load the data into the program
    
    Args:
    - messages_filepath: location of the messages data
    - categories_filepath: location of the categories data
    
    Outputs:
    - df: a pandas DataFrame of the merged datasets
    
    Functionality:
    Reads in the two datasets and merges them based on the shared key: id
    Returns this dataframe.
    '''
    
    # load messages and categories to dataframes, dropping duplicates
    messages = pd.read_csv(messages_filepath).drop_duplicates(subset='id')
    categories = pd.read_csv(categories_filepath).drop_duplicates(subset='id')
    
    # merge the data set
    df = pd.merge(messages, categories, how='inner', on='id', left_index=True)
    
    # return this dataframe
    return df


def clean_data(df):
    '''
    Cleans the dataframe, replacing categories by binary variables
    
    Args:
    - df: pandas DataFrame
    
    Output:
    - df: cleaned pandas DataFrame
    
    Functionality:
    Creates a dataframe of the 36 individual category columns.
    Identifies what categories each row falls into,
    flag 1 or 0 in col depending on if that message fell into category.
    Returns the cleaned dataframe
    '''
    
    # create a dataframe of the 36 individual category columns
    categories = df.set_index('id')['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    category_colnames = [cat_name[0:-2] for cat_name in row]

    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda val: val[-1])

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
       
    # drop the original categories column from `df`
    df.drop(labels='categories', axis=1, inplace=True)  
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    # drop any rows where the message is null
    df = df.dropna(how='any', subset=['message'])
    
    # fill all null values with a zero
    df = df.fillna(value=0)
    
    # reset the index, no need for none-monotonic increasing index
    df = df.reset_index().drop('index', axis=1)
    
    return df

def save_data(df, database_filename):
    '''
    Saves the dataframe to a database of a given name
    
    Args:
    - df: a pandas DataFrame
    Outputs:
    - none
    '''
    engine = create_engine('sqlite:///{}.db'.format(database_filename))
    df.to_sql(database_filename, engine, index=False, if_exists='replace') 


def main():
    '''
    Function that is ran if the file is called directly
     
    Args:
    - none
    Outputs:
    - none
    
    Functionality:
    - loads in the filepaths
    - calls function to load data
    - calls function to clean data
    - calls function to save data
    '''
    
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
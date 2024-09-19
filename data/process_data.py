import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df

def clean_data(df):
    categories = df['categories'].str.split(pat=';', expand=True)
    row = categories.loc[0]
    colnames = []
    for entry in row:
        colnames.append(entry[:-2])
    category_colnames = colnames
    print('Column names:', category_colnames)
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1:]
        categories[column] = categories[column].astype(int)
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    df = df[pd.notnull(df.related)]
    df = df[df['related'] != 2]
    print('Duplicates remaining:', df.duplicated().sum())
    return df
    
def save_data(df, database_filename):
    name = 'sqlite:///../data/' + database_filename
    engine = create_engine(name)
    df.to_sql('messages', engine, index=False,if_exists='replace')


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
        print('Please provide the correct filepaths')


if __name__ == '__main__':
    main()

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
    colnames = [entry[:-2] for entry in row]
    categories.columns = colnames

    for column in categories:
        categories[column] = categories[column].str[-1:].astype(int)

    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    floods = df['floods']
    storm = df['storm']
    earthquake = df['earthquake']
    fire = df['fire']
    
    def check_help_in_message(message):
        return 'help' in message.lower()
    
    df['medical_help'] = (
        (df['aid_related'] | df['medical_products'] | df['water'] |
         df['food'] | df['shelter'] | df['clothing'] |
         df['other_aid'] | df['aid_centers']| df['message'].apply(check_help_in_message))
    ).astype(int)
    columns_to_drop = [
        'request', 'offer', 'aid_related', 'medical_products', 'search_and_rescue', 'security',
        'military', 'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', 'missing_people',
        'refugees', 'death', 'other_aid', 'infrastructure_related', 'transport', 'buildings',
        'electricity', 'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
        'weather_related', 'cold', 'other_weather', 'direct_report'
    ]
    df.drop(columns=columns_to_drop, axis=1, inplace=True)

    
    hurricane_keywords = ['hurricane', 'cyclone', 'typhoon', 'storm']
    tornado_keywords = ['tornado', 'twister', 'funnel cloud']
    landslide_keywords = ['landslide', 'landslides', 'mudslide', 'rockslide','roadblocks','block','road','stone','slide']

    df['hurricane'] = 0
    df['tornado'] = 0
    df['landslides'] = 0

    def check_keywords(message, keywords):
        return any(keyword.lower() in message.lower() for keyword in keywords)

    df['hurricane'] = df['message'].apply(lambda x: 1 if check_keywords(x, hurricane_keywords) else 0)
    df['tornado'] = df['message'].apply(lambda x: 1 if check_keywords(x, tornado_keywords) else 0)
    df['landslides'] = df['message'].apply(lambda x: 1 if check_keywords(x, landslide_keywords) else 0)

    df['related'] = (
        (df['floods'] | df['medical_help'] | df['earthquake'] |
         df['storm'] | df['fire'] | df['landslides'] |
         df['tornado'] | df['hurricane'])
    ).astype(int)

    df.drop_duplicates(inplace=True)
    df = df[pd.notnull(df.related)]
    df = df[df['related'] != 2]
    print('Duplicates remaining:', df.duplicated().sum())

    return df

def save_data(df, database_filename, csv_filename):
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace')

    df.to_csv(csv_filename, index=False)
    print(f'Data saved to database: {database_filename} and CSV file: {csv_filename}')

def main():
    if len(sys.argv) == 5:
        messages_filepath, categories_filepath, database_filepath, csv_filepath = sys.argv[1:]

        print(f'Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}')
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print(f'Saving data...\n    DATABASE: {database_filepath}\n    CSV FILE: {csv_filepath}')
        save_data(df, database_filepath, csv_filepath)

        print('Cleaned data saved to database and CSV file!')

    else:
        print('Please provide the correct filepaths: messages_filepath, categories_filepath, database_filepath, csv_filepath')


if __name__ == '__main__':
    main()

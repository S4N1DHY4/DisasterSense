import pandas as pd
import numpy as np
import re
import warnings
import sys
import pickle

from sqlalchemy import create_engine

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer

from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn import multioutput
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report

import nltk
nltk.download(['punkt_tab', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages', engine)

    X = df['message']
    Y = df.iloc[:, 4:]

    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model_randomforest():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])

    return pipeline

def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    return pipeline

def build_model_randomizedsearch():
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', HashingVectorizer(tokenizer=tokenize, n_features=2**16)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__learning_rate': [0.01, 0.1, 1, 2]
    }
    
    model = RandomizedSearchCV(pipeline, param_distributions=parameters, n_iter=5, cv=3, verbose=2, n_jobs=-1)
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    overall_accuracy = (y_pred == Y_test).mean().mean() * 100
    y_pred = pd.DataFrame(y_pred, columns=Y_test.columns)

    for col in Y_test.columns:
        print('Category feature : {}'.format(col.capitalize()))
        print('.................................................................\n')
        print(classification_report(Y_test[col], y_pred[col]))
        accuracy = (y_pred[col].values == Y_test[col].values).mean().mean() * 100
        print('Accuracy: {0:.1f} %\n'.format(accuracy))

    print('Overall Accuracy: {0:.1f} %'.format(overall_accuracy))
    pass


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)
    pass


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
        print('Please provide the correct filepath')

if __name__ == '__main__':
    main()
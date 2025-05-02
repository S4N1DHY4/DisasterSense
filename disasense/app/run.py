import json
import plotly.graph_objs as gob
import plotly
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from textblob import TextBlob
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
import spacy
from location import extract_locations

nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)


disease_data = pd.read_csv('data/dataset.csv')  
disease_dict = {}

for index, row in disease_data.iterrows():
    disaster_type = row['Disaster'].lower()
    disease = row['Disease']
    prevention_measures = row['Prevention Measures']
    
    if disaster_type not in disease_dict:
        disease_dict[disaster_type] = []
    
    disease_dict[disaster_type].append({
        'disease': disease,
        'prevention_measures': prevention_measures
    })

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


nltk.download('wordnet')

engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql(
    "SELECT * FROM 'messages' LIMIT 7000,1000",
    con=engine
)
df += pd.read_sql(
    "SELECT * FROM 'messages' LIMIT 14500,1000",
    con=engine
)
model = joblib.load("models/classifier.pkl")

@app.route('/')
@app.route('/index')
def index():
    cats = df[df.columns[5:13]]
    cats_counts = cats.mean()*cats.shape[0]
    cats_names = list(cats_counts.index)
    nlarge_counts = cats_counts.nlargest(5)
    nlarge_names = list(nlarge_counts.index)

  
    correlation_matrix = cats.corr()

    heatmap_data = gob.Heatmap(
        z=correlation_matrix.values,  
        x=correlation_matrix.columns,  
        y=correlation_matrix.columns,  
        colorscale='YlOrRd',           
        colorbar=dict(title='Correlation')  
    )

    mean_scores = cats.mean().sort_values(ascending=False)

    heatmap_1d_data = gob.Heatmap(
        z=mean_scores.values.reshape(1,-1),
        x=correlation_matrix.columns,  
        y=['Mean Danger Scores'], 
        colorscale='YlOrRd', 
        colorbar=dict(title='Mean Danger Scores')  
    )

    contour_data = gob.Contour(
        z=mean_scores,
        x=cats.columns,
        y=list(range(cats.shape[0])),
        colorscale='Viridis',
        colorbar=dict(title='Contour Plot')
    )

    graphs = [
        {
            'data': [heatmap_1d_data],

            'layout': {
                'title': '1D Heatmap',
                'xaxis': {
                    'title': 'Category',
                    'tickangle': 35  
                },
                'yaxis': {
                    'title': 'Mean Danger Score',
                    'tickangle': 0,
                    'showticklabels': False
                },
                'paper_bgcolor': '#aff1ff',
                'plot_bgcolor': '#aff1ff'
            }
        },
        {
            'data': [heatmap_data],

            'layout': {
                'title': 'Correlation Heatmap',
                'xaxis': {
                    'title': 'Category',
                    'tickangle': 35  
                },
                'yaxis': {
                    'tickangle': 0,
                    'showticklabels': False
                },
                'paper_bgcolor': '#aff1ff',
                'plot_bgcolor': '#aff1ff'
            }
        },
        {
            'data': [
                Bar(
                    x=cats_names,
                    y=cats_counts,
                    marker=dict(color='red')
                )
            ],

            'layout': {
                'title': 'Distribution of Message categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 35
                },
                'paper_bgcolor': '#aff1ff',
                'plot_bgcolor': '#aff1ff'
            }
        },
        {
            'data': [
                {
                    'x': nlarge_names,
                    'y': nlarge_counts,
                    'type': "scatter",
                    'mode': 'lines+markers', 
                    'line': dict(color='red'),  
                    'marker': dict(color='black')
                }
            ],

            'layout': {
                'title': 'Top message categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                },
                'paper_bgcolor': '#aff1ff',
                'plot_bgcolor': '#aff1ff'
            }
        },
        {
            'data': [contour_data],
            'layout': {
                'title':'Contour Plot',
                'xaxis':{'title':'Category', 'tickangle':35, 'automargin':True},
                'yaxis':{'title':'Samples', 'automargin':True},
                'paper_bgcolor':'#aff1ff',
                'plot_bgcolor':'#aff1ff'
            }
        }
        
    ]
    
    
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


@app.route('/go')
def go():
    query = request.args.get('query', '') 

    locations = extract_locations(query)
    locations_str = ', '.join(locations) if locations else 'No locations found'
    
    classification_labels = model.predict([query])[0] #classifies based on the model (process_data.py, train classifier.py -> on static data)
    classification_results = dict(zip(df.columns[4:], classification_labels))

    classification_results['locations'] = locations_str
    import csv
    with open("../classification_result.csv", "w", newline="") as f:
        w = csv.DictWriter(f, classification_results.keys())
        w.writeheader()
        w.writerow(classification_results)
    
    diseases_info = {}

    for category, classification in classification_results.items():
        if classification == 1:  
            disaster_type = category.lower().replace(" ", "_") 
            if disaster_type in disease_dict:
                diseases_info[category] = disease_dict[disaster_type]

    classification_results.pop('locations')
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results,
        diseases_info=diseases_info
    )

@app.route("/fetch_descriptions", methods=["GET"])
def fetch_descriptions():
    try:
        data = pd.read_csv("data/merged_data.csv")
        descriptions = data["description"].dropna().tolist()
        return jsonify(descriptions)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return jsonify([])

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
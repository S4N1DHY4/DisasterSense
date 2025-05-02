import spacy
import re
from typing import List

nlp = spacy.load("en_core_web_sm")


def clean_text(text: str) -> str:
    """Clean input text by removing special characters and extra whitespace."""
    if not text:
        return ""
    text = re.sub(r'[^\w\s]', ' ', str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_locations(text: str) -> List[str]:
    """Extract location entities from text using spaCy."""
    doc = nlp(clean_text(text))
    locations = []
        
    for ent in doc.ents:
        if ent.label_ in ['GPE', 'LOC']:
            locations.append(ent.text)
        
    return list(set(locations))  # Remove duplicates
        

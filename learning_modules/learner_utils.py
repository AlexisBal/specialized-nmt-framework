import json
from datetime import datetime
from collections import Counter, defaultdict

def find_english_terms_pivot(source_term, pivot_dictionary, source_language):
    """
    Trouve les termes anglais correspondant à un terme source via dictionnaire pivot
    """
    english_terms = []
    
    for entry in pivot_dictionary:
        if source_language == "Hebrew":
            if source_term in entry.get("hebrew_pure", []):
                english_terms.extend(entry.get("english", []))
        elif source_language == "Greek":
            if source_term == entry.get("greek"):
                english_terms.extend(entry.get("english", []))
        elif source_language == "Latin":
            if source_term == entry.get("latin"):
                english_terms.extend(entry.get("english", []))
    
    return english_terms


"""
Accumulator
Accumule les co-occurences pour l'apprentissage
"""
from collections import defaultdict, Counter
cumulative_cooccurrences = defaultdict(Counter)
verse_ref = defaultdict(Counter)

def accumulate_cooccurences_verse(source_language, target_language, source_text, target_text, verse_references):
    global cumulative_cooccurrences
    """
    Accumule les co-occurrences pour un verset
    """
    if (source_language == "English" or source_language == "Italian" or source_language == "French" or source_language == "Spanish" or source_language == "Portuguese"):
        source_text_words = extract_lemmas(source_text, source_language)
    elif source_language == "Chinese":
        source_text_words = extract_chinese_words(source_text)
    else:
        print(f"Langue non prise en charge: voir comment initialiser Spacy pour la lemmatisation")
        return None
    
    if (target_language == "English" or target_language == "Italian" or target_language == "French" or target_language == "Spanish" or target_language == "Portuguese"):
        target_text_words = extract_lemmas(target_text, target_language)
    elif target_language == "Chinese":
        target_text_words = extract_chinese_words(target_text)
    else:
        print(f"Langue non prise en charge: voir comment initialiser Spacy pour la lemmatisation")
        return None
            
    # Accumulation des co-occurrences
    cooc_added = 0
    for source_text_word in set(source_text_words):  # Mots uniques
        for target_text_word in target_text_words:
            cumulative_cooccurrences[source_text_word][target_text_word] += 1
            cooc_added += 1
            verse_ref[f"{source_text_word}_{target_text_word}"].add(verse_references)

    return cumulative_cooccurrences


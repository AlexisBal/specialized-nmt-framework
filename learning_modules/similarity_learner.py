"""
Similarity Learner (pour le chinois exclusivement)
Détecte les variantes de caractère à partir des entrées du dictionnaire de référence, de la longueur et de la similarité pinyin
"""
import json
from datetime import datetime
from collections import Counter, defaultdict
from utils.pinyin_functions import get_pinyin, calculate_similarity
from learning_modules.learner_utils import find_english_terms_pivot


def learn_from_similarity_verse(
    verse_reference,            # référence du verset analysé
    cleaned_cooccurrences,      # issues de l'apprentissage par dictionnaire
    semantic_dictionary,        # {source_term: [target_term_candidates]}
    pinyin_dictionary,          # {source_term: {target_term: pinyin}} pour chinois, {} sinon
    learned_dictionary,         # dictionnaire des termes déjà appris
    source_language,            # "English", "French", etc.
    target_language,            # "Chinese", "Spanish", etc.
    pivot_dictionary=None       # logique de dictionnaire pivot si langues anciennes
):
    """
    Apprend les variantes orthographiques par similarité pinyin
    Fonctionne uniquement pour target_language == "Chinese"
    """
    
    # La fonction est inutile si le target_language n'est pas le chinois
    if target_language != "Chinese":
        return {
            'similarity_verse_learned_terms': [],
            'similarity_verse_updated_terms': [],
            'similarity_retained_target_terms': set(),
            'similarity_retained_source_terms': set(),
            'cleaned_cooccurrences': cleaned_cooccurrences
        }
    
    # Variables de tracking
    learned_terms = 0
    updated_terms = 0
    learned_terms_summary = []
    updated_terms_summary = []
    retained_source_terms = set()
    retained_target_terms = set()
    
    for source_term in cleaned_cooccurrences:
    
        # Récupérer les traductions de référence (avec pivot si nécessaire)
        dict_translations = []

        # Recherche directe
        if source_term.lower() in semantic_dictionary:
            dict_translations.extend(semantic_dictionary[source_term.lower()])
        if f'to {source_term.lower()}' in semantic_dictionary:
            dict_translations.extend(semantic_dictionary[f'to {source_term.lower()}'])

        # Recherche via pivot pour langues anciennes
        if source_language in ["Greek", "Hebrew", "Latin"] and target_language == "Chinese" and pivot_dictionary:
            english_terms = find_english_terms_pivot(source_term, pivot_dictionary, source_language)
            for english_term in english_terms:
                if english_term.lower() in semantic_dictionary:
                    dict_translations.extend(semantic_dictionary[english_term.lower()])
                if f'to {english_term.lower()}' in semantic_dictionary:
                    dict_translations.extend(semantic_dictionary[f'to {english_term.lower()}'])

        # Continuer seulement si on a des traductions
        if dict_translations:
                
            for target_term, cooccurrence_count in cleaned_cooccurrences[source_term].items():
            
                # Filtre longueur
                if len(target_term) < 2:
                    continue
                    
                # Calcul de similarité
                max_similarity = 0
                best_dict_match = None
                
                for dict_translation in dict_translations:
                    similarity = calculate_similarity(
                        target_term, dict_translation,
                        get_pinyin(target_term), get_pinyin(dict_translation)
                    )
                    if similarity == 1.0: # Identification exacte déjà traitée par le dictionnaire en étape 1,
                        continue          # on ne cherche que les éventuelles variantes parce que le terme apparait plusieurs fois dans le verset
                        
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_dict_match = dict_translation
                
                # Seuil adaptatif selon longueur
                threshold = 0.5 if len(target_term) >= 3 else 0.7
                
                if max_similarity > threshold:
                    
                    if source_term in learned_dictionary["source_term_entry"]:
                        # Si le terme est déjà connu du dictionnaire appris, mettre à jour son entrée
                        learned_targets = learned_dictionary["source_term_entry"][source_term]["learned_targets"]
                        primary_found = False
                        for target in learned_targets:
                            if target["target_term"] == target_term:
                                target["confidence_score"] += max_similarity
                                target["is_primary"] = True
                                primary_found = True
                                break
                        
                        if not primary_found:
                            # Ajouter le terme
                            learned_targets.append({
                                    "target_term": target_term,
                                    "confidence_score": max_similarity,
                                    "is_primary": True
                                })

                        learned_dictionary["source_term_entry"][source_term]["last_updated"] = verse_reference
                        updated_terms += 1
                        updated_terms_summary.append(f"{source_term}→{target_term}")
                        action = "updated"
                        
                    else:
                        # Nouveau terme source
                        learned_targets = []
                        # Ajouter le terme au dictionnaire
                        learned_targets.append({
                            "target_term": target_term,
                            "confidence_score": max_similarity,
                            "is_primary": True
                        })
                        
                        # Créer une nouvelle entrée
                        learned_dictionary["source_term_entry"][source_term] = {
                            "learned_targets": learned_targets,
                            "first_learned": verse_reference,
                            "last_updated": verse_reference
                        }
                        learned_terms += 1
                        learned_terms_summary.append(f"{source_term}→{target_term}")
                        action = "learned"
                            
                    learned_dictionary["cumulative_data"]["learning_history"].append({
                        "verse_reference": verse_reference,
                        "source_term": source_term,
                        "action": action,
                        "primary_target": target_term,
                        "learning_method": "similarity",
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    retained_source_terms.add(source_term)
                    retained_target_terms.add(target_term)
                    
    cleaned_cooccurrences = {
        source: {
            target: count for target, count in targets.items()
            if target not in retained_target_terms and
               not any(retained_term in target for retained_term in retained_target_terms) and
               not any(target in retained_term for retained_term in retained_target_terms)
        }
        for source, targets in cleaned_cooccurrences.items()
        if source not in retained_source_terms
    }
    
    return {
        'similarity_verse_learned_terms': learned_terms_summary,
        'similarity_verse_updated_terms': updated_terms_summary,
        'similarity_retained_target_terms': retained_target_terms,
        'similarity_retained_source_terms': retained_source_terms,
        'cleaned_cooccurrences': cleaned_cooccurrences
    }

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


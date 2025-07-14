"""
Dictionary Learner
Apprends les termes par match exact avec le dictionnaire de référence
Fonction learn_from_dictionary fonctionne pour un verset unique
"""

import json
from datetime import datetime
from collections import Counter, defaultdict
from learning_modules.accumulator import accumulate_cooccurences_verse
from data_preparation.chinese_extractor import extract_chinese_words
from data_preparation.lemmatisor import extract_lemmas
from memory_manager.initialisator import create_learned_dictionary
from memory_manager.initialisator import load_dict_pivot
from learning_modules.learner_utils import find_english_terms_pivot

def learn_from_dictionary_verse(
    semantic_dictionary,        # {source_term: [target_term_candidates]}
    verse_reference,            # référence du verset analysé
    corpus_texts,               # fichier json contenant tous les versets du batch
    learned_dictionary,         # dictionnaire des termes déjà appris
    source_language,            # "English", "French", etc.
    target_language,            # "Chinese", "Spanish", etc.
    pivot_dictionary=None       # Dictionnaire pivot si langues anciennes
):
    """
    Analyse les co-occurrences de termes sources et target pour un verset donné
    Si un match exact est trouvé, identifie le terme target comme primary_target et augmente son score de 1
    Si plusieurs termes sont trouvés, le score est augmenté de 0.2
    """
    
    candidates_analyzed = 0       # Nombre de termes analysés dans le texte target
    candidates_filtered = 0       # Nombre de termes écartés dans le texte target
    learned_terms = 0             # Nombre de nouveaux termes appris par match dictionnaire
    updated_terms = 0             # Nombre de termes mis à jour par match dictionnaire
    learned_terms_summary = []    # Collecte les nouveaux termes appris par match dictionnaire
    updated_terms_summary = []    # Collecte les termes mis à jour par match dictionnaire
    retained_source_terms = set() # Collecte les termes appris dans le texte source
    retained_target_terms = set() # Collecte les termes appris dans le texte target
    
    # 1. Générer les cooccurences du verset analysé
    # 1.1. Retrouver les textes source et target pour le verset analysé
    verse = corpus_texts[verse_reference]
    source_text = verse["source_text"]
    target_text = verse["target_text"]
    
    # 1.2. Générer les termes
    if source_language == "Chinese":
        source_terms = extract_chinese_words(source_text)
    else:
        source_terms = extract_lemmas(source_text, source_language)
        
    if target_language == "Chinese":
        target_terms = extract_chinese_words(target_text)
    else:
        target_terms = extract_lemmas(target_text, target_language)
        
    # 1.3. Générer les co-occurrences de termes
    verse_cooccurrences = defaultdict(Counter)
    for source_term in set(source_terms):
        for target_term in target_terms:
            verse_cooccurrences[source_term][target_term] += 1
    
    # 2. Rechercher les matchs exacts dans le dictionnaire de référence
    for source_term in source_terms:
        candidates_analyzed += 1
        
        # 2.1. Vérifier si le terme source existe dans le dictionnaire
        dict_translations = []
        if source_term.lower() in semantic_dictionary:
            dict_translations.extend(semantic_dictionary[source_term.lower()])
        if f'to {source_term.lower()}' in semantic_dictionary:  # pour les verbes anglais
            dict_translations.extend(semantic_dictionary[f'to {source_term.lower()}'])
            
        if source_language in ["Greek", "Hebrew", "Latin"] and target_language == "Chinese":
            if pivot_dictionary:
                english_terms = find_english_terms_pivot(source_term, pivot_dictionary, source_language)
                for english_term in english_terms:
                    if english_term.lower() in semantic_dictionary:
                        dict_translations.extend(semantic_dictionary[english_term.lower()])
                    if f'to {english_term.lower()}' in semantic_dictionary:
                        dict_translations.extend(semantic_dictionary[f'to {english_term.lower()}'])

        # 2.2. Chercher les matches avec les termes target du verset
        dict_matches = []
        for target_term in target_terms:
            if target_term in dict_translations:
                dict_matches.append(target_term) # dict_matches stockes tous les matchs exactes
        
        # 2.3. Ordonner par taille décroissante pour privilégier les termes les plus longs (attestés dans le target verse)
        dict_matches = sorted(dict_matches, key=len, reverse=True)

        # 2.4. Si match trouvé, ajouter au dictionnaire le match le plus long
        if dict_matches:
            primary_target = dict_matches[0] # Le terme le plus long est prioritaire
            target_alternatives = dict_matches[1:] # Les autres termes sont considérés comme des alternatives
            retained_target_terms.add(primary_target)
            
            if source_term in learned_dictionary["source_term_entry"]:
                # Si le terme est déjà connu du dictionnaire appris, mettre à jour son entrée
                learned_targets = learned_dictionary["source_term_entry"][source_term]["learned_targets"]
                primary_found = False
                for target in learned_targets:
                    if target["target_term"] == primary_target:
                        target["confidence_score"] += 1
                        target["is_primary"] = True
                        primary_found = True
                        break
                
                if not primary_found:
                    # Ajouter le terme
                    learned_targets.append({
                            "target_term": primary_target,
                            "confidence_score": 1,
                            "is_primary": True
                        })
                    
                # Se souvenir des alternatives présentes dans le dictionnaire de référence
                for alternative in target_alternatives:
                    alt_found = False
                    for target in learned_targets:
                        if target["target_term"] == alternative:
                            target["confidence_score"] += 0.2
                            alt_found = True
                            break
                    if not alt_found:
                        learned_targets.append({
                            "target_term": alternative,
                            "confidence_score": 0.2,
                            "is_primary": False
                        })

                learned_dictionary["source_term_entry"][source_term]["last_updated"] = verse_reference
                updated_terms += 1
                updated_terms_summary.append(f"{source_term}→{primary_target}")
                action = "updated"
                
            else:
                # Nouveau terme source
                learned_targets = []
                # Ajouter primary target
                learned_targets.append({
                    "target_term": primary_target,
                    "confidence_score": 1,
                    "is_primary": True
                })
                
                # Ajouter les alternatives pour en conserver la mémoire
                for alternative in target_alternatives:
                    learned_targets.append({
                        "target_term": alternative,
                        "confidence_score": 0.2,
                        "is_primary": False
                    })
                    
                # Créer une nouvelle entrée
                learned_dictionary["source_term_entry"][source_term] = {
                    "learned_targets": learned_targets,
                    "first_learned": verse_reference,
                    "last_updated": verse_reference
                }
                learned_terms += 1
                learned_terms_summary.append(f"{source_term}→{primary_target}")
                action = "learned"
                    
            learned_dictionary["cumulative_data"]["learning_history"].append({
                "verse_reference": verse_reference,
                "source_term": source_term,
                "action": action,
                "primary_target": primary_target,
                "alternatives_found": target_alternatives,
                "learning_method": "dictionary",
                "timestamp": datetime.now().isoformat()
            })
            
            retained_source_terms.add(source_term)
    
    # Filtrage des termes consommés pour les étapes suivantes
    cleaned_cooccurrences = {
        source: {
            target: count for target, count in targets.items()
            if target not in retained_target_terms and
               not any(retained_term in target for retained_term in retained_target_terms)
        }
        for source, targets in verse_cooccurrences.items()
        if source not in retained_source_terms or source_terms.count(source) > 1
    }
    
    return {
        'dict_verse_learned_terms': learned_terms_summary,    # Liste des nouveaux termes appris
        'dict_verse_updated_terms': updated_terms_summary,    # Liste des termes mis à jour
        'retained_target_terms': retained_target_terms,       # Set des termes target retenus
        'retained_source_terms': retained_source_terms,       # Set des termes source traités
        'cleaned_cooccurrences': cleaned_cooccurrences,        # Cooccurrences non traitées par le dictionnaire
    }


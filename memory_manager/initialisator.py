"""
Initialisator
Initialise les données en début d'apprentissage.
"""

import json, re
import os
from datetime import datetime
from collections import defaultdict, Counter

def create_learned_dictionary():
    """
    Crée le dictionnaire d'apprentissage.
    """
    return {
        "source_term_entry": {},
        "metadata": {
            "batch_processed": [],
            "verses_processed": 0,
            "creation_date": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "unresolved_proper_nouns": []
        },
        "cumulative_data": {
            "cooccurrences": {},
            "learning_history": []
        }
    }

import re

def parse_source_target_dictionary(source_language, target_language, source_target_dictionary):
    if (source_language == "English" and
        target_language == "Chinese"):
        
        source_target_dictionary_path = source_target_dictionary
        semantic_dict = {}
        pinyin_dict = {}
        
        try:
            with open(source_target_dictionary_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Ignorer commentaires et lignes vides
                    if not line or line.startswith('#'):
                        continue
                        
                    # Parser format: 繁體 简体 [pinyin] /def1/def2/
                    match = re.match(r'^(.+?)\s+(.+?)\s+\[([^\]]+)\]\s+/(.+)/$', line)
                    if not match:
                        continue
                        
                    traditional, simplified, pinyin, definitions = match.groups()
                    def_list = [d.strip().lower() for d in definitions.split('/') if d.strip()]
                    
                    # Ajouter chaque définition anglaise comme clé
                    for definition in def_list:
                        # Nettoyer la définition
                        clean_def = re.sub(r'\([^)]*\)', '', definition).strip()
                        clean_def = clean_def.split(',')[0].strip()
                        
                        if (len(clean_def) >= 2 and
                            clean_def.replace(' ', '').replace('-', '').isalpha()):
                            
                            if clean_def not in semantic_dict:
                                semantic_dict[clean_def] = []
                                pinyin_dict[clean_def] = {}
                            if simplified.strip() not in semantic_dict[clean_def]:
                                semantic_dict[clean_def].append(simplified.strip())
                                pinyin_dict[clean_def][simplified.strip()] = pinyin.strip()
            
            print(f"📚 CC-CEDICT parsé: {len(semantic_dict)} termes anglais")
            return semantic_dict, pinyin_dict
            
        except FileNotFoundError:
            print(f"⚠️ Fichier {source_target_dictionary_path} non trouvé")
            return {}, {}
            
    elif (source_language in ["Greek", "Hebrew", "Latin"] and target_language == "Chinese"):
        # Charger aussi CC-CEDICT pour English→Chinese
        semantic_dict, pinyin_dict = parse_source_target_dictionary("English", "Chinese", source_target_dictionary)
        return semantic_dict, pinyin_dict # Retour à 3 éléments
    
    else:
        print(f"⚠️ Combinaison non supportée: {source_language}-{target_language}, enrichir Initialisator")
        return {}, {}

def load_dict_pivot(source_language):
    """
    Charge le dictionnaire pivot pour les langues anciennes vers anglais
    """
    pivot_files = {
        "Greek": "input/dictionaries/gre_eng.json",
        "Hebrew": "input/dictionaries/heb_eng.json",
        "Latin": "input/dictionaries/lat_eng.json"
    }
    
    if source_language not in pivot_files:
        return None
        
    try:
        with open(pivot_files[source_language], 'r', encoding='utf-8') as f:
            pivot_dict = json.load(f)
        print(f"📚 Dictionnaire pivot {source_language}→English chargé: {len(pivot_dict)} entrées")
        return pivot_dict
    except FileNotFoundError:
        print(f"⚠️ Fichier pivot {pivot_files[source_language]} non trouvé")
        return None

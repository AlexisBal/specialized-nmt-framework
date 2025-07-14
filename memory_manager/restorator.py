"""
Restorator
Charge la mémoire des données déjà apprises stockées dans le dictionnaire appris
Appelé au début de chaque batch
"""

import json
import os
from collections import defaultdict, Counter


def restore_learned_dictionary(learned_dictionary_path):
    """
    Restaure le dictionnaire d'apprentissage sauvegardé par memorisator.
    
    Args:
        learned_dictionary_path: Chemin du fichier de sauvegarde
    
    Returns:
        tuple: (learned_dictionary, cumulative_cooccurrences, verses_processed)
    """
    try:
        # Charger le dictionnaire
        with open(learned_dictionary_path, 'r', encoding='utf-8') as f:
            learned_dictionary = json.load(f)
        
        # Restaurer les co-occurrences
        cumulative_cooccurrences = defaultdict(Counter)
        verses_processed = 0
        
        if 'cumulative_data' in learned_dictionary:
            # Reconstituer les co-occurrences
            saved_cooc = learned_dictionary['cumulative_data'].get('cooccurrences', {})
            for source_term, target_term_counts in saved_cooc.items():
                cumulative_cooccurrences[source_term].update(target_term_counts)
            
            # Récupérer le compteur de versets
            verses_processed = learned_dictionary['cumulative_data'].get('verses_processed', 0)
        
        print(f"🔄 Dictionnaire restauré: {len(learned_dictionary.get('source_term_entry', {}))} termes, {verses_processed} versets")
        
        return learned_dictionary, cumulative_cooccurrences, verses_processed
        
    except FileNotFoundError:
        print(f"📝 Aucun dictionnaire trouvé, utiliser initialisator.py")
        return None, None, None

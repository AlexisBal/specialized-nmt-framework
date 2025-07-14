"""
Memorisator
Sauvegarde les données apprises après chaque batch
"""

import json
import os
from datetime import datetime
from collections import defaultdict, Counter


def save_batch_learning(
    learned_dictionary,
    cumulative_cooccurrences,
    source_text,
    target_text,
    batch_number,
    verses_processed,
    learned_dictionary_path
):
    """
    Sauvegarde de la progression d'apprentissage pour un batch donné.
    
    Args:
        learned_dictionary: Dictionnaire des termes appris
        cumulative_cooccurrences: defaultdict(Counter) des co-occurrences
        verses_processed: Nombre de versets traités
        source_text: texte source
        target_text : texte target
        batch_number: Numéro de batch
        learned_dictionary_path: Chemin du fichier de sauvegarde du dictionnaire d'apprentissage 
    
    Returns:
        bool: True si sauvegarde réussie
    """
    try:
        # Mettre à jour métadonnées
        if batch_number not in learned_dictionary["metadata"]["batch_processed"]:
            learned_dictionary["metadata"]["batch_processed"].append(batch_number)
        
        learned_dictionary["metadata"]["last_updated"] = datetime.now().isoformat()
        learned_dictionary["metadata"]["verses_processed"] = verses_processed
        
        # Récupérer l'historique existant
        existing_history = learned_dictionary.get("cumulative_data", {}).get("learning_history", [])
        
        # Sauvegarder les co-occurences importantes
        learned_dictionary["cumulative_data"] = {
            "cooccurrences": {
                source_term: dict(target_term_counts)
                for source_term, target_term_counts in cumulative_cooccurrences.items()
            },
            "verses_processed": verses_processed,
            "learning_history": existing_history
        }
        
        # Ajouter entrée à l'historique
        learned_dictionary["cumulative_data"]["learning_history"].append({
            "batch_number": batch_number,
            "timestamp": datetime.now().isoformat(),
            "total_terms": len(learned_dictionary["source_term_entry"]),
            "verses_processed": verses_processed
        })
        
        # Sauvegarder
        with open(learned_dictionary_path, 'w', encoding='utf-8') as f:
            json.dump(learned_dictionary, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Progression sauvegardée: {len(learned_dictionary['source_term_entry'])} termes, {verses_processed} versets")
        return True
        
    except Exception as e:
        print(f"❌ Erreur sauvegarde: {e}")
        return False
    
def backup_previous_version(learned_dictionary_path, backup_suffix="_backup"):
    """
    Crée une sauvegarde du fichier existant avant écrasement.
    
    Args:
        learned_dictionary_path: Chemin du dictionnaire de termes appris
        backup_suffix: Suffixe pour le fichier de backup
    
    Returns:
        str: Chemin du fichier de backup créé ou None
    """
    try:
        if os.path.exists(learned_dictionary_path):
            backup_file = learned_dictionary_path.replace('.json', f'{backup_suffix}.json')
            with open(learned_dictionary_path, 'r', encoding='utf-8') as src:
                with open(backup_file, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
            return backup_file
        return None
    except Exception as e:
        print(f"⚠️ Erreur backup: {e}")
        return None

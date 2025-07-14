"""
Sauvegarde les métadonnées d'entraînement dans l'historique JSON dédié.
"""

import os
import json, re
from datetime import datetime

from tuner.translator import get_mbart_language_code


def save_training_metadata(batch_number, model_path, output_model, source_text, target_text,
                          source_language, target_language, training_metrics):
    """
    Sauvegarde les métadonnées d'entraînement dans l'historique JSON dédié.
    
    Args:
        batch_number: Numéro du batch
        model_path: Modèle de base utilisé
        output_model: Modèle fine-tuné produit
        source_text: Texte source
        target_text: Texte cible
        source_language: Langue source
        target_language: Langue cible
        training_metrics: Métriques calculées
    """
    
    # Chemin du fichier historique
    history_file = "tuner/fine_tuning_history.json"
    os.makedirs("tuner", exist_ok=True)
    
    # Préparer les données du batch actuel
    batch_data = {
        "batch_id": batch_number,
        "timestamp": datetime.now().isoformat(),
        "model_info": {
            "base_model": model_path,
            "output_model": output_model,
            "source_language": source_language,
            "target_language": target_language,
            "language_codes": {
                "source": get_mbart_language_code(source_language),
                "target": get_mbart_language_code(target_language)
            }
        },
        "training_metrics": training_metrics
    }
    
    # Charger l'historique existant ou créer nouveau
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            history_data = {
                "metadata": {
                    "total_fine_tunings": 0,
                    "created": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat()
                },
                "fine_tunings": []
            }
    else:
        history_data = {
            "metadata": {
                "total_fine_tunings": 0,
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            },
            "fine_tunings": []
        }
    
    # Ajouter le nouveau batch
    history_data["fine_tunings"].append(batch_data)
    history_data["metadata"]["total_fine_tunings"] = len(history_data["fine_tunings"])
    history_data["metadata"]["last_updated"] = datetime.now().isoformat()
    
    # Sauvegarder
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)
    
    print(f"📋 Historique mis à jour: {history_file} ({history_data['metadata']['total_fine_tunings']} fine-tunings)")

"""
Cleaner
Nettoie les modèles après chaque batch pour sauver de l'espace mémoire
"""

import gc
import torch
import os
import shutil
from datetime import datetime

def cleanup_intermediate_files(model_path=None):
    """
    Nettoie les fichiers intermédiaires du fine-tuning.
    
    Args:
        model_path: Chemin du modèle de base à nettoyer (optionnel)
    """

    # 1. Nettoyer les modèles temporaires 
    models_dir = "outputs/models"
    if os.path.exists(models_dir):
        model_folders = [f for f in os.listdir(models_dir) if f.startswith("batch_")]
        if len(model_folders) > 2:  # Garder seulement les 2 plus récents
            model_folders.sort()
            to_delete = model_folders[:-2]
            
            for folder in to_delete:
                folder_path = os.path.join(models_dir, folder)
                try:
                    shutil.rmtree(folder_path)
                    print(f"   🗑️ Supprimé: {folder}")
                except Exception as e:
                    print(f"   ⚠️ Erreur: {e}")
    
    # 2. Nettoyer les caches
        #if torch.cuda.is_available():
            #torch.cuda.empty_cache()
       # elif torch.backends.mps.is_available():
         #   torch.mps.empty_cache()
    gc.collect()
    
    # 3. Nettoyer le cache HuggingFace
    cleanup_huggingface_cache()
    
    # 4. Nettoyer les logs anciens
    cleanup_logs()
    
    print("   ✅ Nettoyage terminé")

def cleanup_huggingface_cache():
    """Nettoie le cache HuggingFace si trop volumineux"""
    import os
    from pathlib import Path
    
    cache_dir = Path.home() / ".cache" / "huggingface"
    if cache_dir.exists():
        try:
            # Calculer taille
            total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
            size_gb = total_size / (1024**3)
            
            if size_gb > 10:  # Si > 10 GB
                print(f"   🗑️ Cache HF volumineux ({size_gb:.1f} GB), nettoyage...")
                shutil.rmtree(cache_dir, ignore_errors=True)
                print(f"   ✅ Cache HF nettoyé")
        except Exception as e:
            print(f"   ⚠️ Erreur nettoyage cache HF: {e}")

def cleanup_logs():
    """Nettoie les anciens logs"""
    logs_dir = "outputs/logs"
    if os.path.exists(logs_dir):
        # Garder seulement les logs des 7 derniers jours
        cutoff = datetime.now().timestamp() - (7 * 24 * 3600)
        for log_file in os.listdir(logs_dir):
            file_path = os.path.join(logs_dir, log_file)
            if os.path.getmtime(file_path) < cutoff:
                os.remove(file_path)

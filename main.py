"""
Fichier principal main
Entrées : source_text, target_text, source_language, target_language, source_target_dictionary, book_range

1. Préparation et organisation des textes : dossier text_organizer. Fonction principale : batch_creator/prepare_texts
Charge les textes du book_range
Analyse les données: identifie des péricopes par rupture de style, puis décompose les données en créant des batches
Crée un fichier json contenant les textes organisés par batchs, péricopes, versets, avec deux champs vides destinés à accueillir les traductions pré et post tunage du modèle.
Retourne un corpus_texts organisé stocké dans le json.

2. Apprentissage par batches (orchestrateur : learner_modules.learner.py)
2.1. Apprentissage initial sur les données réelles : source_texts et target_texts. Apprentissage par dictionnaire et similarité verset par verset, par dominance et détection de noms propres batch par batch si demandé
2.2. Tuning d'un modèle (orchestrateur : tuner.finetuner.py) si demandé
2.3. Rapport d'apprentissange et rapport de tuning par batch (si apprentissage ou tuning)
4. Cumule les données globales sur le range étudié
"""

import argparse
import json
import os
from datetime import datetime
from typing import List, Dict, Any
import gc
import torch
import shutil
from pathlib import Path

from text_organizer.batch_creator import prepare_texts
from learning_modules.learner import run_learning
from memory_manager.initialisator import create_learned_dictionary, parse_source_target_dictionary, load_dict_pivot
from memory_manager.restorator import restore_learned_dictionary
from memory_manager.memorisator import save_batch_learning

def parse_arguments():
    """Parse les arguments essentiels."""
    parser = argparse.ArgumentParser(description='Système de fine-tuning biblique')
    
    parser.add_argument('--book_range', type=str, required=True,
                       help='Range de livres (ex: "1" ou "1-3")')
    parser.add_argument('--source_language', type=str, default='English',
                       help='Langue source (English, French, Italian, Portuguese, Spanish, Chinese)')
    parser.add_argument('--target_language', type=str, default='Chinese',
                       help='Langue cible (English, French, Italian, Portuguese, Spanish, Chinese)')
    parser.add_argument('--source_text', type=str, default='input/bibles/EnglishNIVBible.xml')
    parser.add_argument('--target_text', type=str, default='input/bibles/ChineseUnion2010Bible.xml')
    parser.add_argument('--source_target_dictionary', type=str, default='input/dictionaries/cedict_ts.u8',
                       help='Chemin vers le dictionnaire de référence pour apprentissage')
    parser.add_argument('--learning', action='store_true', default=False,
                       help='Activer apprentissage terminologique')
    parser.add_argument('--no-learning', action='store_true', default=False,
                       help='Désactiver apprentissage terminologique')
    parser.add_argument('--tuning', action='store_true', default=False,
                       help='Activer fine-tuning des modèles')
    parser.add_argument('--no-tuning', action='store_true', default=False,
                       help='Désactiver fine-tuning des modèles')
    parser.add_argument('--base_model', type=str, default='facebook/mbart-large-50-many-to-many-mmt',
                       help='Modèle de base mBART')
    parser.add_argument('--learned_dictionary_path', type=str, default='dictionaries/learned_dictionary.json',
                       help='Chemin vers le dictionnaire des termes appris')
    
    args = parser.parse_args()
    
    # Gérer les flags booléens correctement
    if args.no_learning:
        args.learning = False
    elif not args.learning and not args.no_learning:
        args.learning = True  # Valeur par défaut
    
    if args.no_tuning:
        args.tuning = False
    elif not args.tuning and not args.no_tuning:
        args.tuning = True  # Valeur par défaut
        
    return args

def load_dictionaries(args):
    """Charge les dictionnaires nécessaires à l'apprentissage."""
    
    # Dictionnaire sémantique initial
    pivot_config = None
    if args.source_target_dictionary:
        semantic_dictionary, pinyin_dictionary = parse_source_target_dictionary(
            source_language=args.source_language,
            target_language=args.target_language,
            source_target_dictionary=args.source_target_dictionary
        )
        # Charger le pivot automatiquement si langues anciennes → chinois
        if args.source_language in ["Greek", "Hebrew", "Latin"] and args.target_language == "Chinese":
            pivot_dictionary = load_dict_pivot(args.source_language)
        else:
            pivot_dictionary = None
    else:
        semantic_dictionary, pinyin_dictionary, pivot_dictionary = {}, {}, None
    
    # Dictionnaire des termes appris (persistant entre les runs)
    restored_data = restore_learned_dictionary(args.learned_dictionary_path)
    
    if restored_data[0] is not None:
        learned_dictionary, cumulative_cooccurrences, verses_processed = restored_data
    else:
        print("📝 Création nouveau dictionnaire appris")
        learned_dictionary = create_learned_dictionary()
        cumulative_cooccurrences = {}
        verses_processed = 0
    
    return semantic_dictionary, learned_dictionary, pinyin_dictionary, pivot_dictionary

def save_learned_dictionary(learned_dictionary, filepath):
    """Sauvegarde le dictionnaire appris en utilisant memorisator."""
    try:
        # Utiliser la fonction existante memorisator
        success = save_batch_learning(
            learned_dictionary=learned_dictionary,
            cumulative_cooccurrences={},  # Sera géré par learner
            source_text="",  # Sera géré par learner
            target_text="",  # Sera géré par learner
            batch_number=0,  # Sera géré par learner
            verses_processed=learned_dictionary.get('metadata', {}).get('verses_processed', 0),
            learned_dictionary_path=filepath
        )
        if success:
            print(f"💾 Dictionnaire appris sauvegardé: {filepath}")
    except Exception as e:
        print(f"❌ Erreur sauvegarde dictionnaire appris: {e}")

def print_global_summary(global_stats, args):
    print(f"\n✅ TRAITEMENT TERMINÉ")
    print(f"📚 {args.book_range} | {args.source_language}→{args.target_language} | {global_stats['total_batches']} batches")
    if args.tuning and global_stats.get('final_model'):
        print(f"🎯 Modèle final: {global_stats['final_model']}")

def main(book_range=None, source_language=None, target_language=None,
         source_text=None, target_text=None, Learning=True, Tuning=True,
         source_target_dictionary=None, base_model=None, learned_dictionary_path=None):
    """Fonction principale"""
    
    # Si des paramètres sont passés directement, les utiliser
    if book_range is not None:
        # Créer un objet args simple
        class Args:
            pass
        
        args = Args()
        args.book_range = book_range
        args.source_language = source_language or 'English'
        args.target_language = target_language or 'Chinese'
        args.source_text = source_text or 'input/bibles/EnglishNIVBible.xml'
        args.target_text = target_text or 'input/bibles/ChineseUnion2010Bible.xml'
        args.source_target_dictionary = source_target_dictionary
        args.learning = Learning if Learning is not None else True
        args.tuning = Tuning if Tuning is not None else True
        args.base_model = base_model or ('facebook/mbart-large-50-many-to-many-mmt' if Tuning else None)
        args.learned_dictionary_path = learned_dictionary_path or 'dictionaries/learned_dictionary.json'
        args.pivot_dictionary = None
    else:
        # Sinon, parser les arguments de ligne de commande
        args = parse_arguments()
    print(f"🚀 Démarrage fine-tuning biblique")
    print(f"📖 Livres: {args.book_range}")
    print(f"📄 Sources: {args.source_text} → {args.target_text}")
    print(f"🌐 Langues: {args.source_language} → {args.target_language}")
    print(f"📚 Dictionnaire: {args.source_target_dictionary or 'Aucun'}")
    print(f"🧠 Learning activé: {args.learning}")
    print(f"🎯 Tuning activé: {args.tuning}")
    print(f"🤖 Modèle de base: {args.base_model}")
    
    # Validation des paramètres
    if not args.learning and not args.tuning:
        print("❌ Erreur: Au moins Learning ou Tuning doit être activé")
        return 1
        
    if args.tuning:
        print(f"🤖 Modèle de base: {args.base_model}")
    else:
        print(f"🤖 Modèle de base: Non requis (tuning désactivé)")
    
    # Validation des paramètres
    if not args.learning and not args.tuning:
        print(f" Besoin d'au moins une fonction activée")
    
    try:
        # Étape 1: Préparation des données
        print(f"\n📋 ÉTAPE 1: Préparation des textes")
        corpus_texts = prepare_texts(
            source_text=args.source_text,
            target_text=args.target_text,
            book_range=args.book_range
        )
        
        # Étape 2: Chargement des dictionnaires
        print(f"\n📚 ÉTAPE 2: Chargement des dictionnaires")
        if args.learning:
            semantic_dictionary, learned_dictionary, pinyin_dictionary, pivot_dictionary = load_dictionaries(args)
        else:
            semantic_dictionary, learned_dictionary, pinyin_dictionary, pivot_dictionary = {}, {}, {}, None
            print("⏭️ Learning désactivé, dictionnaires non chargés")
        
        # Étape 3: Apprentissage
        print(f"\n🧠 ÉTAPE 3: Pipeline d'apprentissage")
        global_stats = run_learning(
            book_range=args.book_range,
            semantic_dictionary=semantic_dictionary,
            learned_dictionary=learned_dictionary,
            pinyin_dictionary=pinyin_dictionary,
            pivot_dictionary=pivot_dictionary,
            source_language=args.source_language,
            target_language=args.target_language,
            base_model=args.base_model,
            learning_enabled=args.learning,
            tuning_enabled=args.tuning
        )
        
        # Étape 4: Rapport global
        print_global_summary(global_stats, args)
        
        print(f"\n✅ Fine-tuning terminé avec succès!")
        
        # Garbage collection Python
        gc.collect()

        # Caches PyTorch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        torch.clear_autocast_cache()

        # Cache pip temporaire
        pip_cache = Path.home() / ".cache" / "pip"
        if pip_cache.exists():
            shutil.rmtree(pip_cache, ignore_errors=True)

        # Variables d'environnement tokenizers
        import os
        if "TOKENIZERS_PARALLELISM" in os.environ:
            del os.environ["TOKENIZERS_PARALLELISM"]

        print("🗑️ Tous les caches nettoyés - système restauré")
        return 0
        
    except Exception as e:
        print(f"\n❌ Erreur fatale: {e}")
        
        # Garbage collection Python
        gc.collect()

        # Caches PyTorch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        torch.clear_autocast_cache()

        # Cache pip temporaire
        pip_cache = Path.home() / ".cache" / "pip"
        if pip_cache.exists():
            shutil.rmtree(pip_cache, ignore_errors=True)

        # Variables d'environnement tokenizers
        import os
        if "TOKENIZERS_PARALLELISM" in os.environ:
            del os.environ["TOKENIZERS_PARALLELISM"]

        print("🗑️ Tous les caches nettoyés - système restauré")
        return 1

    
if __name__ == "__main__":
    exit(main())

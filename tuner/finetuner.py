"""
Fine-tuning de modèles mBART pour la traduction biblique, verset par verset pour un batch.
"""

import os, glob
import time
import shutil
from datetime import datetime
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import torch
from tuner.training_utils import load_model_for_training, preserve_chinese_punctuation, get_mbart_language_code
from tuner.translator import cleanup_memory
from tuner.metrics_calculator import calculate_training_metrics
from tuner.metadata_manager import save_training_metadata
from memory_manager.cleaner import cleanup_intermediate_files

def fine_tune_batch(model_path, source_text, target_text, batch_number,
                   source_language, target_language, corpus_texts, batch_reference):
    """
    Fine-tune un modèle mBART sur un batch complet de données bibliques.
    Tente d'abord l'optimisation automatique des hyperparamètres, puis fallback sur les paramètres manuels.
    
    Args:
        model_path: Chemin vers le modèle de base
        source_text: Texte source pour l'ensemble du batch
        target_text: Texte cible pour l'ensemble du batch
        batch_number: Numéro du batch pour nommage
        source_language: Nom de langue source ("English", "Chinese", etc.)
        target_language: Nom de langue cible ("English", "Chinese", etc.)
        
    Returns:
        str: Chemin du modèle fine-tuné (ou None si échec)
    """
    DISABLE_OPTUNA = False
    
    if not source_text or not target_text:
        print("❌ Textes source ou cible vides")
        return None
    
    # Générer le chemin de sortie
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/models/batch_{batch_number}_{timestamp}"
    
    print(f"   Fine-tuning batch: {source_language} → {target_language}")
    print(f"   Modèle: {model_path.split('/')[-1]} → {output_path}")
    
    cleanup_memory()
    
    # Charger le modèle
    training_device = torch.device("cpu")
    model, tokenizer, device = load_model_for_training(model_path)
    if model is None:
        return None
        
    # Dataset pour Trainer
    class VerseDataset(torch.utils.data.Dataset):
        def __init__(self, source_verses, target_verses, tokenizer, source_language, target_language, device):
            self.examples = []
            
            src_code = get_mbart_language_code(source_language)
            tgt_code = get_mbart_language_code(target_language)
            
            for src_verse, tgt_verse in zip(source_verses, target_verses):
                # Configurer les langues
                tokenizer.src_lang = src_code
                tokenizer.tgt_lang = tgt_code
                
                # Post-traitement chinois si nécessaire
                if target_language == "Chinese":
                    tgt_verse = preserve_chinese_punctuation(tgt_verse)
                
                # Tokeniser chaque verset individuellement
                inputs = tokenizer(
                    src_verse,
                    max_length=512,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )
                
                labels = tokenizer(
                    tgt_verse,
                    max_length=512,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )
                
                self.examples.append({
                    'input_ids': inputs['input_ids'].squeeze(0).to(device),
                    'attention_mask': inputs['attention_mask'].squeeze(0).to(device),
                    'labels': labels['input_ids'].squeeze(0).to(device)
                })
        
        def __len__(self):
            return len(self.examples)
        
        def __getitem__(self, idx):
            return self.examples[idx]
    
    try:

        # Récupérer les versets individuels du batch
        batch_verses = []
        for verse_id, verse_data in corpus_texts.items():
            if (verse_id != "batch_data" and
                isinstance(verse_data, dict) and
                verse_data.get("batch_reference") == batch_reference):
                batch_verses.append({
                    'source': verse_data['source_text'],
                    'target': verse_data['target_text']
                })

        # Diviser les versets (85% train, 15% eval)
        split_idx = int(len(batch_verses) * 0.85)
        train_verses = batch_verses[:split_idx]
        eval_verses = batch_verses[split_idx:]

        
        # Préparer les listes de versets
        train_source_verses = [v['source'] for v in train_verses]
        train_target_verses = [v['target'] for v in train_verses]
        eval_source_verses = [v['source'] for v in eval_verses]
        eval_target_verses = [v['target'] for v in eval_verses]

        # Créer les datasets par verset
        train_dataset = VerseDataset(
            train_source_verses, train_target_verses,
            tokenizer, source_language, target_language, device
        )
        eval_dataset = VerseDataset(
            eval_source_verses, eval_target_verses,
            tokenizer, source_language, target_language, device
        )
        
        print(f"📊 Dataset créé: {len(train_dataset)} versets train, {len(eval_dataset)} versets eval")
        
        # Créer le répertoire de sortie
        os.makedirs(output_path, exist_ok=True)
        
        # Sauvegarder la liste des fichiers avant optimisation (pour supprimer les données d'essai d'optimisation)
        files_before = set(os.listdir(output_path)) if os.path.exists(output_path) else set()
        trainer = None
        final_model = None
        training_time = 0
        # Tenter l'optimisation automatique des hyperparamètres
        try:
            import optuna
            use_auto_hp = True
            print("📊 Tentative d'optimisation automatique des hyperparamètres avec Optuna...")
        except ImportError:
            use_auto_hp = False
            print("⚠️  Optuna non disponible, utilisation des paramètres manuels")
        
        if use_auto_hp and not DISABLE_OPTUNA:
            try:
                base_model, base_tokenizer, device = load_model_for_training(model_path)
                # Fonction de création du modèle pour l'optimisation (réinitialiser seulement les poids pour ne pas recharger le modèle à chaque fois)
                def model_init(trial=None):
                    import copy
                    model_copy = copy.deepcopy(base_model)
                    return model_copy
                
                # Définir l'espace de recherche pour Optuna
                def optuna_hp_space(trial):
                    return {
                        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
                        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 8),
                        "warmup_steps": trial.suggest_int("warmup_steps", 0, 20),
                        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
                    }
                
                # Arguments de base pour l'optimisation
                training_args = TrainingArguments(
                    output_dir=output_path,
                    per_device_train_batch_size=1,
                    per_device_eval_batch_size=1,
                    eval_strategy="steps",
                    eval_steps=5,
                    save_steps=5,
                    logging_steps=2,
                    save_total_limit=2,
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_loss",
                    greater_is_better=False,
                    remove_unused_columns=False,
                    save_safetensors=False,
                    dataloader_num_workers=0,
                    dataloader_pin_memory=False,
                )
                
                # Créer le trainer pour l'optimisation
                trainer = Trainer(
                    model=None,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    processing_class=tokenizer,
                    model_init=model_init,
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
                )
                
                print(f"🔍 Recherche des meilleurs hyperparamètres...")
                start_time = time.time()
                
                # Lancer la recherche automatique (nombre réduit de trials pour votre cas)
                best_trials = trainer.hyperparameter_search(
                    direction="minimize",
                    backend="optuna",
                    hp_space=optuna_hp_space,
                    n_trials=5,  # Réduit pour éviter des temps trop longs
                    compute_objective=lambda metrics: metrics.get("eval_loss", float('inf'))
                )
                
                training_time = time.time() - start_time
                print(f"✅ Optimisation terminée en {training_time:.1f}s")
                print(f"📈 Meilleurs hyperparamètres: {best_trials.hyperparameters}")
                
                best_params = best_trials.hyperparameters
                model, tokenizer, device = load_model_for_training(model_path)
                
                # Arguments d'entraînement avec les meilleurs paramètres
                best_training_args = TrainingArguments(
                    output_dir=output_path,
                    per_device_train_batch_size=1,
                    per_device_eval_batch_size=1,
                    eval_strategy="steps",
                    eval_steps=5,
                    save_steps=5,
                    logging_steps=2,
                    save_total_limit=2,
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_loss",
                    greater_is_better=False,
                    remove_unused_columns=False,
                    save_safetensors=False,
                    dataloader_num_workers=0,
                    dataloader_pin_memory=False,
                    **best_params  # Utiliser les paramètres optimisés
                )
                
                # Créer le trainer final avec les meilleurs paramètres
                final_trainer = Trainer(
                    model=model,
                    args=best_training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    processing_class=tokenizer,
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
                )
                
                print(f"🎯 Entraînement final avec les meilleurs paramètres...")
                start_final_time = time.time()
                final_trainer.train()
                final_training_time = time.time() - start_final_time
                training_time = final_training_time  # Remplacer le temps d'optimisation
                
                final_model = final_trainer.model
                trainer = final_trainer  # Pour la suite du code qui utilise trainer.state
                
                # Nettoyage des optimizer.pt d'Optuna
                cleanup_patterns = [
                    f"{output_path}/**/optimizer.pt",     # Dans le dossier du batch
                    f"{output_path}/**/pytorch_model.bin", # Dans le dossier du batch
                    "outputs/models/**/optimizer.pt",      # Tous les anciens aussi
                    "outputs/models/**/pytorch_model.bin"  # Tous les anciens aussi
                ]
                for pattern in cleanup_patterns:
                    files_to_clean = glob.glob(pattern, recursive=True)
                    for file_path in files_to_clean:
                        try:
                            os.remove(file_path)
                            print(f"🧹 Checkpoint Optuna supprimé: {file_path}")
                        except Exception as e:
                            print(f"⚠️ Erreur suppression {file_path}: {e}")
                
            except Exception as e:
                print(f"⚠️  Erreur dans l'optimisation automatique: {e}")
                print("🔄 Basculement vers les paramètres manuels...")
                use_auto_hp = False
                trainer = None
        
        # Fallback : utiliser les paramètres manuels actuels
        if not use_auto_hp or trainer is None:
            # Arguments d'entraînement manuels (vos paramètres actuels)
            training_args = TrainingArguments(
                output_dir=output_path,
                num_train_epochs=5,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                learning_rate=3e-5,
                warmup_steps=5,
                eval_strategy="steps",
                eval_steps=5,
                save_steps=5,
                logging_steps=2,
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                remove_unused_columns=False,
                save_safetensors=False,
                weight_decay=0.01,
                dataloader_pin_memory=False,
            )
            
            # Créer le trainer avec paramètres manuels
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=tokenizer,
                callbacks=[
                    EarlyStoppingCallback(early_stopping_patience=2)
                ]
            )
            
            print(f"📚 Training sur batch complet avec paramètres manuels...")
            start_time = time.time()
            trainer.train()
            training_time = time.time() - start_time
            final_model = trainer.model
            
        if trainer is None:
            raise Exception("Impossible de créer le trainer")
        if final_model is None:
            final_model = trainer.model
        
        # Calculer les métriques de training
        training_metrics = calculate_training_metrics(
            train_history=trainer.state.log_history,
            training_time=training_time,
            source_language=source_language,
            target_language=target_language
        )
        
        # Sauvegarder le modèle final
        try:
            # Essai de safetensors
            final_model.save_pretrained(
                output_path,
                safe_serialization=True,
                max_shard_size="5GB"  # Évite les fichiers trop volumineux
            )
            print(f"✅ Modèle sauvé avec safetensors")
        except Exception as e:            # Fallback : sauvegarde PyTorch complète
            print(f"⚠️ Erreur safetensors: {e}")
            print(f"🔄 Basculement vers sauvegarde PyTorch complète...")
            final_model.save_pretrained(
                output_path,
                safe_serialization=False,  # Force la sauvegarde complète
                max_shard_size="5GB"
            )
            print(f"✅ Modèle sauvé avec PyTorch format")

        # Tokenizer (toujours sauver)
        tokenizer.save_pretrained(output_path)
        files_after_model_save = set(os.listdir(output_path)) # pour nettoyage
        legitimate_files = files_after_model_save - files_before
        
        # Sauvegarder les métadonnées
        save_training_metadata(
            batch_number=batch_number,
            model_path=model_path,
            output_model=output_path,
            source_text=source_text,
            target_text=target_text,
            source_language=source_language,
            target_language=target_language,
            training_metrics=training_metrics
        )
        
        # Nettoyer les fichiers temporaires créés pendant l'optimisation
        if use_auto_hp:
            all_files_final = set(os.listdir(output_path))
            temp_files = all_files_final - files_before - legitimate_files
            for temp_file in temp_files:
                temp_path = os.path.join(output_path, temp_file)
                try:
                    if os.path.isdir(temp_path):
                        shutil.rmtree(temp_path)
                    else:
                        os.remove(temp_path)
                    print(f"🧹 Supprimé fichier temporaire: {temp_file}")
                except Exception as cleanup_error:
                    print(f"⚠️  Impossible de supprimer {temp_file}: {cleanup_error}")
                                
        print(f"✅ Fine-tuning terminé: {output_path}")
        return output_path, training_metrics
        
    except Exception as e:
        print(f"❌ Erreur pendant le fine-tuning: {e}")
        return None
    
    finally:
        cleanup_memory()

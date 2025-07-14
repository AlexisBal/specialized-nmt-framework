"""
Translator
Traducteur utilisant mBART pour la traduction des textes bibliques.
"""

import os
import torch
import gc
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from tuner.training_utils import preserve_chinese_punctuation, get_mbart_language_code, get_training_device

# Configuration pour réduire les warnings
os.environ["TOKENIZERS_PARALLELISM"] = "warning"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


def load_model_and_tokenizer(model_path, force_device=None):
    """
    Charge le modèle et le tokenizer mBART depuis un chemin donné.
    
    Args:
        model_path: Chemin vers le modèle (local ou HuggingFace)
        
    Returns:
        tuple: (model, tokenizer, device)
    """
    if force_device:
        device = force_device
    else:
        device = torch.device("cpu")
    
    try:
        # Charger le modèle et tokenizer
        model = MBartForConditionalGeneration.from_pretrained(model_path)
        tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
        
        # Déplacer le modèle sur le device approprié
        model = model.to(device)
        model.eval()  # Mode évaluation pour l'inférence
        
        print(f"✅ Modèle chargé: {model_path.split('/')[-1]} sur {device}")
        return model, tokenizer, device
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle {model_path}: {e}")
        return None, None, None


def cleanup_memory():
    """Nettoie la mémoire GPU/CPU"""
    gc.collect()

def translate(model_path, source_text, source_language, target_language,
              tokenizer=None, model=None, device=None):
    """
    Traduit un texte donné en utilisant mBART.
    
    Args:
        model_path: Chemin vers le modèle mBART (utilisé seulement si model=None)
        source_text: Texte à traduire
        source_language: Nom de langue source ("English", "Chinese", etc.)
        target_language: Nom de langue cible ("English", "Chinese", etc.)
        tokenizer: Tokenizer optionnel (si None, sera chargé)
        model: Modèle optionnel (si None, sera chargé depuis model_path)
        device: Device optionnel (si None, sera détecté)
        
    Returns:
        str: Texte traduit avec ponctuation appropriée
    """
    
    if not source_text or not source_text.strip():
        print("⚠️ Texte source vide")
        return ""
    
    # Convertir les noms de langues en codes mBART
    src_code = get_mbart_language_code(source_language)
    tgt_code = get_mbart_language_code(target_language)
    
    # Charger si pas déjà fourni
    load_model_needed = (model is None)
    
    if load_model_needed:
        model, tok, device = load_model_and_tokenizer(model_path)
        if model is None:
            return ""
        
        # Utiliser le tokenizer fourni ou celui chargé
        if tokenizer is None:
            tokenizer = tok
    
    try:
        # Préparer les inputs
        inputs = tokenizer(
            source_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Configurer les langues
        tokenizer.src_lang = src_code
        tokenizer.tgt_lang = tgt_code
                
        # Génération
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id[tgt_code],
                max_new_tokens=512,
                min_length=10,
                num_beams=5,
                do_sample=False,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                early_stopping=True,          
                pad_token_id=tokenizer.pad_token_id
            )

        # Décoder le résultat
        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        # Post-traitement spécifique au chinois
        if target_language == "Chinese":
            translation = preserve_chinese_punctuation(translation)
        
        print(f"🔤 Traduction: {len(source_text)} → {len(translation)} chars")
        return translation
        
    except Exception as e:
        print(f"❌ Erreur lors de la traduction: {e}")
        return ""
    
    finally:
        if load_model_needed:
            cleanup_memory()


def translate_batch_texts(model_path, source_texts, source_language, target_language):
    """
    Traduit une liste de textes pour un batch
    """
    
    if not source_texts:
        return []
    
    # Charger le modèle UNE SEULE FOIS
    model, tokenizer, device = load_model_and_tokenizer(model_path)
    if model is None or tokenizer is None or device is None:
        print(f"❌ Échec chargement modèle: model={model}, tokenizer={tokenizer}, device={device}")
        return [""] * len(source_texts)
    
    translations = []
    
    try:
        cleanup_memory()
        
        for text in source_texts:
            if not text or not text.strip():
                translations.append("")
                continue
                
            # Passer le modèle déjà chargé
            translation = translate(
                model_path=model_path,  # Pas utilisé car model fourni
                source_text=text,
                source_language=source_language,
                target_language=target_language,
                tokenizer=tokenizer,    # Réutilise le tokenizer
                model=model,           # Réutilise le modèle
                device=device          # Réutilise le device
            )
            translations.append(translation)
        
        return translations
        
    except Exception as e:
        print(f"❌ Erreur lors de la traduction par batch: {e}")
        return [""] * len(source_texts)
    
    finally:
        cleanup_memory()  # Nettoie à la fin du batch

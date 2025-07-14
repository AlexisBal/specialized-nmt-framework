"""
Fonctions utils pour le training
"""

import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

def load_model_for_training(model_path):
    """
    Charge le modèle et tokenizer pour le fine-tuning.
    """
    device = torch.device("cpu")
    
    try:
        model = MBartForConditionalGeneration.from_pretrained(model_path)
        # Essayer de configurer le tokenizer pour préserver la ponctuation chinoise
        try:
            tokenizer = MBart50TokenizerFast.from_pretrained(
                model_path,
                sp_model_kwargs={
                    "enable_sampling": False,
                    "alpha": 0.0,
                    "nbest_size": 1
                }
            )

        except Exception as e:
            tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
            
        model = model.to(device)
        model.train()
        
        print(f"✅ Modèle chargé pour fine-tuning: {model_path.split('/')[-1]} sur {device}")
        return model, tokenizer, device
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle {model_path}: {e}")
        return None, None, None
        
def preserve_chinese_punctuation(text):
    """
    Préserve explicitement la ponctuation chinoise traditionnelle
    en gérant correctement les guillemets ouvrants et fermants.
    """
    # Mapping simple ponctuation
    simple_map = {
        ',': '，',  # Virgule
        '.': '。',  # Point
        '?': '？',  # Point d'interrogation
        '!': '！',  # Point d'exclamation
        ':': '：',  # Deux-points
        ';': '；',  # Point-virgule
        ' ': '',    # Supprime les espaces
    }
    
    result = text
    
    # Appliquer mappings simples
    for western, chinese in simple_map.items():
        result = result.replace(western, chinese)
    
    # Gérer les guillemets avec booléens
    in_double_quote = False
    in_single_quote = False
    chars = list(result)
    
    for i, char in enumerate(chars):
        if char == '"':
            if not in_double_quote:
                chars[i] = '「'  # Ouvrant
                in_double_quote = True
            else:
                chars[i] = '」'  # Fermant
                in_double_quote = False
        elif char == "'":
            if not in_single_quote:
                chars[i] = '『'  # Ouvrant simple
                in_single_quote = True
            else:
                chars[i] = '』'  # Fermant simple
                in_single_quote = False
    
    return ''.join(chars)

def get_mbart_language_code(language_name):
    """
    Convertit un nom de langue en code mBART.
    
    Args:
        language_name: Nom de langue ("English", "Chinese", etc.)
        
    Returns:
        str: Code mBART correspondant ("en_XX", "zh_CN", etc.)
    """
    language_mapping = {
        "English": "en_XX",
        "French": "fr_XX",
        "Italian": "it_IT",
        "German": "de_DE",
        "Spanish": "es_XX",
        "Portuguese": "pt_XX",
        "Hebrew": "he_IL",    # hébreu moderne, langue la plus proche de l'hébreu biblique
        "Greek": "en_XX",     # Grec non supporté, fallback sur anglais, plus souple pour l'apprentissage de langues nouvelles. Le macédonien ou le bulgare sont des langues slaves, pas pertinentes
        "Latin": "it_IT",      # Italien, langue la plus proche du latin, malgré l'absence de déclinaisons
        "Chinese": "zh_CN"
    }
    
    code = language_mapping.get(language_name)
    if code is None:
        print(f"⚠️ Langue non supportée: {language_name}. Utilisation de 'en_XX' par défaut.")
        return "en_XX"
    
    return code

def get_training_device():
    """Retourne le device optimal pour l'entraînement"""
    return torch.device("cpu")

def get_metrics_device():
    """Retourne le device optimal pour les métriques (toujours CPU sur Mac)"""
    return torch.device("cpu")

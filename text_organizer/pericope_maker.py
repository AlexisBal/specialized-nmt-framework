"""
Pericope Maker - Analyse les ruptures stylistiques pour découper le texte en péricopes

Ce module utilise Sentence Transformers pour analyser les similarités stylistiques
entre versets consécutifs et identifier les ruptures narratives qui délimitent
les péricopes bibliques.

Fonction principale:
- analyze_style_breaks(): Analyse le corpus complet et retourne les péricopes
  au format "book_chapter_verse-chapter_verse" (ex: "1_1_1-2_24")
"""

import numpy as np
from typing import Dict, List
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def detect_narrative_breaks(embeddings: np.ndarray, verse_references: List[str], min_pericope_size=3) -> List[str]:
    """Détecte les ruptures narratives avec seuils multiples et taille minimale de péricope."""
    
    breaks = [0]
    similarities = []
    for i in range(len(embeddings) - 1):
        similarity = cosine_similarity(
            embeddings[i].reshape(1, -1),
            embeddings[i + 1].reshape(1, -1)
        )[0, 0]
        similarities.append(similarity)
    major_threshold, minor_threshold = auto_detect_thresholds(similarities)
    print(f"🎯 Seuils automatiques: majeur={major_threshold:.3f}, mineur={minor_threshold:.3f}")
    
    # Détecter les ruptures
    for i in range(len(similarities)):
        similarity = similarities[i]
        current_pericope_size = i + 1 - breaks[-1]
        if similarity < major_threshold and current_pericope_size >= min_pericope_size:
            breaks.append(i + 1)
        elif similarity < minor_threshold and current_pericope_size >= min_pericope_size:
            breaks.append(i + 1)
    
    breaks.append(len(embeddings))
    breaks = merge_short_pericopes(breaks, min_pericope_size)
    
    # Créer les péricopes
    pericopes = []
    for i in range(len(breaks) - 1):
        start_idx = breaks[i]
        end_idx = breaks[i + 1] - 1
        
        start_ref = verse_references[start_idx]
        end_ref = verse_references[end_idx]
        
        end_parts = end_ref.split('_')
        end_chapter_verse = f"{end_parts[1]}_{end_parts[2]}"
        
        pericope = f"{start_ref}-{end_chapter_verse}"
        pericopes.append(pericope)
    
    print(f"📜 {len(pericopes)} péricopes créées")
    return pericopes


def make_pericopes(corpus_texts: Dict) -> List[str]:
    """Analyse les ruptures stylistiques sur le book_range indiqué dans le texte source = source_texts."""
    
    source_texts = corpus_texts['source_texts']
    verse_references = corpus_texts['verse_references']
    
    # Sentence Transformer
    model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
    embeddings = model.encode(source_texts)
    
    # Détecter ruptures
    pericopes = detect_narrative_breaks(embeddings, verse_references)
    
    return pericopes

def auto_detect_thresholds(similarities):
    """Calcule des seuils adaptatifs basés sur la distribution des similarités."""
    
    # Statistiques de base
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    
    # Seuils automatiques
    major_threshold = mean_sim - 2 * std_sim  # Ruptures majeures (2σ)
    minor_threshold = mean_sim - 1 * std_sim  # Ruptures mineures (1σ)
    
    # Bornes de sécurité
    major_threshold = max(0.0, major_threshold)
    minor_threshold = max(major_threshold + 0.01, minor_threshold)
    
    return major_threshold, minor_threshold

def merge_short_pericopes(breaks: List[int], min_pericope_size: int) -> List[int]:
    """
    Fusionne les péricopes trop courtes avec leurs voisines.
    """
    if len(breaks) <= 2:
        return breaks
    
    merged_breaks = [breaks[0]]  # Garder le premier break (0)
    
    for i in range(1, len(breaks) - 1):  # Exclure le dernier break
        current_size = breaks[i] - merged_breaks[-1]
        next_size = breaks[i + 1] - breaks[i]
        
        # Si la péricope actuelle est trop courte
        if current_size < min_pericope_size:
            # Ne pas ajouter ce break, fusionner avec la suivante
            continue
        # Si la péricope suivante sera trop courte
        elif next_size < min_pericope_size and i < len(breaks) - 2:
            # Ne pas ajouter ce break, fusionner la suivante avec l'actuelle
            continue
        else:
            # Péricope de taille acceptable
            merged_breaks.append(breaks[i])
    
    merged_breaks.append(breaks[-1])  # Garder le dernier break
    
    return merged_breaks

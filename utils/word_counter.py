import re
from collections import defaultdict, Counter


def lat_count_occurrences(text):
    occurrences = defaultdict(int)
    words = text.split()  # Split simple sur les espaces
    for word in words:
        # Nettoyer la ponctuation de base
        word = re.sub(r'^[^\w]+|[^\w]+$', '', word)  # Enlever ponctuation début/fin
        if word:  # Si le mot n'est pas vide après nettoyage
            occurrences[word] += 1
    return occurrences

def chi_count_occurrences(text):
    """Compte les occurrences de tous les segments chinois dans un texte"""
    occurrences = defaultdict(int)
    
    # Compter les segments de longueur 1 à 4
    for i in range(len(text)):
        for length in [1, 2, 3, 4]:
            if i + length <= len(text):
                segment = text[i:i+length]
                occurrences[segment] += 1
    
    return occurrences


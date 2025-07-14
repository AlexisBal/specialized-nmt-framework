"""
Batch Creator
Stucture le book_range en batchs, péricopes, charge et organise les textes source et target
"""

import json
import os
from typing import List, Dict, Any

from text_organizer.pericope_maker import make_pericopes
from text_organizer.load_texts import load_source_target_texts
from text_organizer.reorganize_texts import reorganize_by_batches


def prepare_texts(source_text: str, target_text: str, book_range: str) -> Dict[str, Any]:
    """Processus pour traiter les données du book_range"""

    # 1. Analyser la structure des livres
    books_structure = analyze_books_structure(source_text, book_range)
    
    # 2. Charger le corpus des textes source et target
    corpus_texts = load_source_target_texts(book_range, source_text, target_text)
    
    # 3. Décomposer le texte en péricopes
    pericopes = make_pericopes(corpus_texts) # Retourne une liste de péricopes (liste de str) pour les livres de book_range
    
    # 4. Créer des batches depuis les péricopes identifiées 
    batch_list = create_batches(books_structure, pericopes)

    # 5. Ré-organiser corpus_texts par batches et par versets, pour le fine-tuning
    batch_list = reorganize_by_batches(corpus_texts, batch_list)
    
    return {
    'success': True,
    'batches_processed': len(batch_list),
    'results': batch_list
    }

def analyze_books_structure(source_text: str, book_range: str) -> Dict[int, List[int]]:
    """
    Analyse la structure XML et retourne le nombre de versets par chapitre pour chaque livre.
    Entrée : source_text, book_range
    Sortie du type (pour les trois premiers chapitres de la Genèse par exemple) : Structure livres 1-3: {1: [31, 25, 24, 26, 32, 22, 24, 22, 29, 32, 32, 20, 18, 24, 21, 16, 27, 33, 38, 18, 34, 24, 20, 67, 34, 35, 46, 22, 35, 43, 55, 32, 20, 31, 29, 43, 36, 30, 23, 23, 57, 38, 34, 34, 28, 34, 31, 22, 33, 26], 2: [22, 25, 22, 31, 23, 30, 25, 32, 35, 29, 10, 51, 22, 31, 27, 36, 16, 27, 25, 26, 36, 31, 33, 18, 40, 37, 21, 43, 46, 38, 18, 35, 23, 35, 35, 38, 29, 31, 43, 38], 3: [17, 16, 17, 35, 19, 30, 38, 36, 24, 20, 47, 8, 59, 57, 33, 34, 16, 30, 37, 27, 24, 33, 44, 23, 55, 46, 34]}
    """
    
    import xml.etree.ElementTree as ET
    
    # Parser le book_range
    if '-' in book_range:
        start_book, end_book = map(int, book_range.split('-'))
        book_numbers = list(range(start_book, end_book + 1))
    else:
        book_numbers = [int(book_range)]
    
    tree = ET.parse(source_text)
    root = tree.getroot()
    
    books_structure = {}
    for book_num in book_numbers:
        book_element = root.find(f".//book[@number='{book_num}']")
        if book_element is None:
            continue
            
        chapter_verse_counts = []
        for chapter in book_element.findall('chapter[@number]'):
            verse_count = len(chapter.findall('verse[@number]'))
            chapter_verse_counts.append(verse_count)
        
        books_structure[book_num] = chapter_verse_counts
        print(f"📖 Livre {book_num}: {len(chapter_verse_counts)} chapitres, {sum(chapter_verse_counts)} versets total")
    
    return books_structure

def create_batches(books_structure, pericopes) -> List[Dict[str, Any]]:
    """Crée les batches pour un range de livre book_range en se basant sur les péricopes identifiées par texts_preparation/pericope_maker, ou par batch de 90 versets."""

    batches = []
    
    for book_num, chapter_verse_counts in books_structure.items():
        book_pericopes = [p for p in pericopes if p.startswith(f"{book_num}_")]  # Identifie les péricopes d'un livre donné
        
        # Si aucune péricope détectée, créer des batches arbitraires de 90 versets
        if not book_pericopes:
            print(f"⚠️ Aucune péricope détectée pour livre {book_num}, création batches arbitraires de 90 versets")
            
            total_verses = sum(chapter_verse_counts)
            
            # Si le livre fait moins de 90 versets, créer un seul batch pour tout le livre
            if total_verses <= 90:
                print(f"📖 Livre {book_num} court ({total_verses} versets), création d'un batch unique")
                last_chapter = len(chapter_verse_counts)
                last_verse = chapter_verse_counts[-1]
                
                batch = {
                    'batch_number': f"book{book_num}_batch1",
                    'batch_reference': f"{book_num}_1_1-{last_chapter}_{last_verse}",
                    'book_number': book_num,
                    'pericopes': [f"{book_num}_1_1-{last_chapter}_{last_verse}"],
                    'total_verses': total_verses
                }
                batches.append(batch)
                continue
            
            # Sinon, créer des batches de 90 versets maximum
            batch_number = 1
            verse_counter = 0
            start_chapter = 1
            start_verse = 1
            
            for chapter_idx, verses_in_chapter in enumerate(chapter_verse_counts, 1):
                for verse_idx in range(1, verses_in_chapter + 1):
                    verse_counter += 1
                    
                    # Si on atteint 90 versets OU si c'est le dernier verset du livre
                    if verse_counter == 90 or (chapter_idx == len(chapter_verse_counts) and verse_idx == verses_in_chapter):
                        # Créer le batch
                        end_chapter = chapter_idx
                        end_verse = verse_idx
                        
                        batch = {
                            'batch_number': f"book{book_num}_batch{batch_number}",
                            'batch_reference': f"{book_num}_{start_chapter}_{start_verse}-{end_chapter}_{end_verse}",
                            'book_number': book_num,
                            'pericopes': [f"{book_num}_{start_chapter}_{start_verse}-{end_chapter}_{end_verse}"],
                            'total_verses': verse_counter
                        }
                        batches.append(batch)
                        print(f"batch arbitraire créé : Livre : {book_num}, Numéro de batch : {batch_number}, {verse_counter} versets")
                        
                        # Préparer le batch suivant
                        batch_number += 1
                        verse_counter = 0
                        start_chapter = chapter_idx
                        start_verse = verse_idx + 1
                        
                        # Si on est à la fin du chapitre, passer au suivant
                        if verse_idx == verses_in_chapter:
                            start_chapter = chapter_idx + 1
                            start_verse = 1
            
            continue  # Passer au livre suivant
        
        # Traitement avec péricopes détectées
        current_batch_verses = 0
        current_batch_pericopes = []
        batch_number = 1
        
        for pericope in book_pericopes: # Parcourt toutes les péricopes d'un livre donné
            pericope_verses = calculate_verses_in_pericope(pericope, books_structure) # Calcule le nombre de versets de la péricope
            
            remaining_verses = sum(calculate_verses_in_pericope(p, books_structure)
                     for p in book_pericopes[book_pericopes.index(pericope)+1:]) # Pour éviter que le dernier batch soit trop petit
                     
            if (current_batch_verses + pericope_verses > 95 and current_batch_pericopes and remaining_verses >= 50):
                # Finaliser le batch actuel
                batch = create_batch_from_pericopes(book_num, batch_number, current_batch_pericopes, books_structure)
                batches.append(batch)
                print(f"batch créé : Livre : {book_num}, Numéro de batch : {batch_number}, Péricopes : {current_batch_pericopes}")
                # Commencer un nouveau batch
                current_batch_pericopes = [pericope]
                current_batch_verses = pericope_verses
                batch_number += 1
                
            else:
                # Ajouter au batch actuel
                current_batch_pericopes.append(pericope)
                current_batch_verses += pericope_verses
        
        # Finaliser le dernier batch
        if current_batch_pericopes:
            batch = create_batch_from_pericopes(book_num, batch_number, current_batch_pericopes, books_structure)
            batches.append(batch)
            print(f"batch créé : Livre : {book_num}, Numéro de batch : {batch_number}, Péricopes : {current_batch_pericopes}")
    
    return batches

def create_batch_from_pericopes(book_num, batch_number, pericopes, books_structure) -> Dict[str, Any]:
    """Crée un batch à partir d'une liste de péricopes et de la structure des livres donnés."""
    
    first_pericope = pericopes[0]
    last_pericope = pericopes[-1]
    
    # Extraire le début de la première péricope et la fin de la dernière péricope
    first_start = first_pericope.split('-')[0]  # ex: "1_1_1"
    last_end = last_pericope.split('-')[1]      # ex: "2_24"
    
    batch_reference = f"{first_start}-{last_end}"  # ex: "1_1_1-2_24"
    
    return {
        'batch_number': f"book{book_num}_batch{batch_number}",
        'batch_reference': batch_reference,
        'book_number': book_num,
        'pericopes': pericopes,
        'total_verses': sum(calculate_verses_in_pericope(p, books_structure) for p in pericopes)
    }

def calculate_verses_in_pericope(pericope: str, books_structure: Dict[int, List[int]]) -> int:
    """Calcule le nombre de versets dans une péricope en consultant books_structure."""
    
    # Parse "1_1_1-2_24"
    start_str, end_str = pericope.split('-')
    
    # Parse start: "1_1_1" -> book=1, chapter=1, verse=1
    book_num, start_chapter, start_verse = map(int, start_str.split('_'))
    
    # Parse end: "2_24" -> chapter=2, verse=24
    end_chapter, end_verse = map(int, end_str.split('_'))
    
    total_verses = 0
    
    if start_chapter == end_chapter:
        # Même chapitre
        total_verses = end_verse - start_verse + 1
    else:
        # Multi-chapitres
        # Versets du premier chapitre
        total_verses += books_structure[book_num][start_chapter - 1] - start_verse + 1
        
        # Chapitres intermédiaires complets
        for ch in range(start_chapter + 1, end_chapter):
            total_verses += books_structure[book_num][ch - 1]
        
        # Versets du dernier chapitre
        total_verses += end_verse
    
    return total_verses


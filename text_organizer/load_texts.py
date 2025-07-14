import xml.etree.ElementTree as ET
import json, os
from typing import List, Dict, Any
from opencc import OpenCC
cc = OpenCC('t2s')

def load_source_target_texts(book_range: str, source_text: str, target_text: str) -> Dict[str, List[str]]:
    """Charge tous les textes source/target pour le book_range complet."""
    
    # Parser le book_range
    if '-' in book_range:
        start_book, end_book = map(int, book_range.split('-'))
        book_numbers = list(range(start_book, end_book + 1))
    else:
        book_numbers = [int(book_range)]
    
    # Charger les XML
    source_tree = ET.parse(source_text)
    target_tree = ET.parse(target_text)
    source_root = source_tree.getroot()
    target_root = target_tree.getroot()
    
    all_source_texts = []
    all_target_texts = []
    verse_references = []
    
    for book_num in book_numbers:
        # Trouver les livres dans les deux XML
        source_book = source_root.find(f".//book[@number='{book_num}']")
        target_book = target_root.find(f".//book[@number='{book_num}']")
        
        if source_book is None or target_book is None:
            print(f"⚠️ Livre {book_num} non trouvé")
            continue
        
        # Parcourir tous les chapitres et versets
        for chapter in source_book.findall('chapter[@number]'):
            chapter_num = chapter.get('number')
            target_chapter = target_book.find(f"chapter[@number='{chapter_num}']")
            
            if target_chapter is None:
                continue
                
            for verse in chapter.findall('verse[@number]'):
                verse_num = verse.get('number')
                target_verse = target_chapter.find(f"verse[@number='{verse_num}']")
                
                if target_verse is None or not verse.text or not target_verse.text:
                    continue
                
                # Ajouter les textes et les références
                all_source_texts.append(verse.text.strip())
                target_text_clean = target_verse.text.strip().replace(' ', '')
                target_text_simplified = cc.convert(target_text_clean)
                all_target_texts.append(target_text_simplified)
                verse_references.append(f"{book_num}_{chapter_num}_{verse_num}")
                verse_lookup = {}
                for i, verse_ref in enumerate(verse_references):
                    verse_lookup[verse_ref] = {
                        'source_text': all_source_texts[i],
                        'target_text': all_target_texts[i],
                        'index': i
                    }
    
    # Sauvegarder corpus complet
    corpus_texts = {
        'book_range': book_range,
        'total_verses': len(all_source_texts),
        'verse_references': verse_references,
        'source_texts': all_source_texts,
        'target_texts': all_target_texts,
        'verse_lookup': verse_lookup
    }
    
    os.makedirs("outputs/translations", exist_ok=True)
    with open(f"outputs/translations/corpus_{book_range.replace('-', '_')}.json", 'w', encoding='utf-8') as f:
        json.dump(corpus_texts, f, ensure_ascii=False, indent=2)
    
    print(f"📚 Corpus chargé: {len(all_source_texts)} versets")
    return corpus_texts

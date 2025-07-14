import json
import os, re
from typing import List, Dict, Any

def reorganize_by_batches(corpus_texts, batch_list):
    """Réorganise corpus_texts par batches."""
    
    verse_lookup = corpus_texts['verse_lookup']
    organized_batches = []
    
    for batch_idx, batch in enumerate(batch_list):
        batch_texts = {
            'batch_number': batch['batch_number'],
            'batch_reference': batch['batch_reference'],
            'pericope_count': len(batch['pericopes']),
            'verse_count': batch['total_verses'],
            'verses': []
        }
        
        # Extraire tous les versets de toutes les péricopes du batch
        for pericope_idx, pericope in enumerate(batch['pericopes']):
            start_ref, end_ref = pericope.split('-')
            start_idx = corpus_texts['verse_references'].index(start_ref)
            book_num = start_ref.split('_')[0]
            end_full_ref = f"{book_num}_{end_ref}"
            try:
                end_idx = corpus_texts['verse_references'].index(end_full_ref)
            except ValueError:
                print(f"Référence non trouvée: {end_full_ref}")
                continue
            
            for i in range(start_idx, end_idx + 1):
                verse_ref = corpus_texts['verse_references'][i]
                corpus_texts['verse_lookup'][verse_ref]['batch_number'] = batch_idx + 1 # Pour commencer à 1 et non à 0
                corpus_texts['verse_lookup'][verse_ref]['batch_reference'] = batch['batch_reference']
                corpus_texts['verse_lookup'][verse_ref]['pericope_number'] = pericope_idx + 1 # Pour commencer à 1 et non à 0
                corpus_texts['verse_lookup'][verse_ref]['pericope_reference'] = pericope
                batch_texts['verses'].append({
                    'verse_reference': verse_ref,
                    'source_text': corpus_texts['source_texts'][i],
                    'target_text': corpus_texts['target_texts'][i]
                })

        organized_batches.append(batch_texts)

    for verse_ref in corpus_texts['verse_lookup']:
        corpus_texts['verse_lookup'][verse_ref]['pretuned_target_text'] = ""
        corpus_texts['verse_lookup'][verse_ref]['posttuned_target_text'] = ""
    os.makedirs("outputs/translations", exist_ok=True)
    with open(f"outputs/translations/corpus_{corpus_texts['book_range'].replace('-', '_')}.json", 'w', encoding='utf-8') as f:
        json.dump(corpus_texts['verse_lookup'], f, ensure_ascii=False, indent=2)
    corpus_texts = organized_batches
    return corpus_texts

def reorganize_verses_from_index(batch_text: str, verse_count: int) -> list:
    """
    Redivise un texte batch traduit en versets individuels selon les séparateurs </s>.
    
    Args:
        batch_text: Texte batch traduit contenant les séparateurs </s>
        verse_count: Nombre de versets attendus
        
    Returns:
        list: Liste des versets individuels
    """
    return re.findall(r'<v>(.*?)</v>', batch_text)

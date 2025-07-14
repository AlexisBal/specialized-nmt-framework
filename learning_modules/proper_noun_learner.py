"""
Proper Noun Learner
Apprend les noms propres par une analyse batch des textes complets, en filtrant les noms propres non résolus
"""

import re
import difflib
from datetime import datetime
from collections import defaultdict, Counter
from utils.pinyin_functions import get_pinyin, calculate_name_similarity, calculate_similarity
from utils.word_counter import lat_count_occurrences,chi_count_occurrences
from data_preparation.chinese_extractor import _extracted_proper_nouns

def learn_from_proper_noun_batch(
    batch_source_text,          # Texte source complet du batch
    batch_target_text,          # Texte cible complet du batch
    cooccurrences_batch,        # Cooccurrences avant dominance (pour l'hébreu)
    cooccurrences_proper_batch, # Cooccurrences restantes après dominance
    batch_reference,            # Référence du batch
    learned_dictionary,         # Dictionnaire des termes appris
    source_language,            # Langue source
    target_language,            # Langue cible
    pinyin_dictionary=None      # Dictionnaire pinyin optionnel
):
    """
    Détecte et apprend les noms propres sur un batch complet
    
    Étapes:
    1. Identifier les noms propres potentiels (termes avec majuscules)
    2. Filtrer selon les cooccurrences restantes et égalité de comptage
    3. Supprimer la pollution par exclusion de caractères fréquents
    4. Résoudre les conflits par similarité pinyin
    5. Réattribuer les perdants à leurs alternatives
    """

    # Variables de tracking
    learned_terms_summary = []
    updated_terms_summary = []
    retained_source_terms = set()
    retained_target_terms = set()
    unresolved_proper_nouns = []
    
    # ===== ÉTAPE 1: IDENTIFIER LES NOMS PROPRES POTENTIELS =====
    
    # Compter les occurrences dans le texte source selon la langue
    if source_language == "Hebrew":
        # Utiliser Stanza hbo pour cette fonction particulière de détection des noms propres
        proper_candidates = detect_hebrew_proper_nouns(batch_source_text)
    elif source_language == "Chinese":
        # Utiliser les noms propres détectés par Stanza
        global _extracted_proper_nouns
        # Compter seulement les occurrences des noms propres détectés par Stanza
        source_occurrences = chi_count_occurrences(batch_source_text)
        proper_candidates = {
            word: count for word, count in source_occurrences.items()
            if word in _extracted_proper_nouns
        }
        print(f"🔍 Candidats avant filtrage et traitement: {proper_candidates}")
    elif source_language == "Greek":
        # Réinitialiser et construire la liste des noms propres pour ce batch
        from data_preparation.lemmatisor import reset_greek_proper_nouns
        reset_greek_proper_nouns()
        
        # Lemmatiser tout le texte batch pour collecter les noms propres
        from data_preparation.lemmatisor import extract_lemmas, _cltk_greek_proper_nouns
        extract_lemmas(batch_source_text, "Greek")
        
        proper_candidates = _cltk_greek_proper_nouns.copy()
        print(f"🔍 Candidats avant filtrage et traitement: {proper_candidates}")
        
    else:
        source_occurrences = lat_count_occurrences(batch_source_text)
        # Garder seulement les termes avec majuscule initiale pour les autres langues
        proper_candidates = {
            word: count for word, count in source_occurrences.items()
            if word[0].isupper()
        }


    # ===== ÉTAPE 2: FILTRAGE =====
    
    # 2.1. Ne conserver que ceux encore dans les cooccurrences non traités avant, sauf pour l'hébreu qui bypasse le filtre dominance (dictionnnaire solide)
    candidates_in_cooccurrences = {}
    
    if source_language == "Hebrew":
        for candidate, source_count in proper_candidates.items():
            if candidate in cooccurrences_batch:
                candidates_in_cooccurrences[candidate] = {
                    'source_count': source_count,
                    'target_counts': cooccurrences_batch[candidate]
                }
    else:
        for candidate, source_count in proper_candidates.items():
            if candidate in cooccurrences_proper_batch:
                candidates_in_cooccurrences[candidate] = {
                    'source_count': source_count,
                    'target_counts': cooccurrences_proper_batch[candidate]
                }

    # 2.2. Filtrer les target_terms avec égalité de comptage
    filtered_candidates = {}
    for candidate, data in candidates_in_cooccurrences.items():
        source_count = data['source_count']
        target_counts = data['target_counts']

        # Ne garder que les target_terms avec count == source_count
        matching_targets = {
            target: count for target, count in target_counts.items()
            if count == source_count
        }
        if matching_targets:
            filtered_candidates[candidate] = {
                'source_count': source_count,
                'target_counts': matching_targets
            }
            
    if not filtered_candidates:
        return {
            'proper_noun_learned_terms': learned_terms_summary,
            'proper_noun_updated_terms': updated_terms_summary,
            'proper_noun_retained_terms': retained_source_terms,
            'proper_noun_unresolved': unresolved_proper_nouns
        }
    
    # ===== ÉTAPE 3: SUPPRESSION DE LA POLLUTION POUR LE CHINOIS =====
    
    # Compter la fréquence des caractères target dans tous les candidats
    if target_language == "Chinese":
        char_frequency = defaultdict(int)
        # Caractères génériques qui ne sont jamais des préfixes/suffixes de noms propres
        generic_chars = {'的', '了', '與', '和', '或', '但', '因', '從', '到', '向', '於', '為', '由', '及', '而'
        }
            
        # Filtrer les target_terms contenant des caractères génériques
        cleaned_candidates = {}
        for candidate, data in filtered_candidates.items():
            clean_targets = {}
            for target_term, count in data['target_counts'].items():
                # Vérifier si le terme commence ou finit par des caractères génériques
                has_generic = (target_term[0] in generic_chars or target_term[-1] in generic_chars) if target_term else True
                
                if not has_generic:
                    if len(target_term) >= 2:
                        clean_targets[target_term] = count
                    
            
            if clean_targets:
                cleaned_candidates[candidate] = {
                    'source_count': data['source_count'],
                    'target_counts': clean_targets
                }
    
        filtered_candidates = cleaned_candidates
    else:
        pass # Pour les langues non-chinoises, pas de filtrage pollution
    
    
    # ===== ÉTAPE 4: RÉSOLUTION DES CONFLITS PAR SIMILARITÉ DE PRONONCIATION =====
    # Charger les dictionnaires une seule fois si nécessaire
    greek_dict = None
    hebrew_dict = None
    uroman_instance = None
    
    if source_language == "Greek":
        try:
            import json
            with open('input/dictionaries/gre_eng.json', 'r', encoding='utf-8') as f:
                greek_dict = json.load(f)
        except:
            pass
    elif source_language == "Hebrew":
        try:
            import json
            with open('input/dictionaries/heb_eng.json', 'r', encoding='utf-8') as f:
                hebrew_dict = json.load(f)
        except:
            pass
        # Initialiser uroman une seule fois
        try:
            import uroman as ur
            uroman_instance = ur.Uroman()
        except:
            pass
    else:
        pass  # Pour les langues non-chinoises
            
    # Créer une liste des associations candidat->target avec similarité
    candidate_associations = []
    for source_candidate, data in filtered_candidates.items():
        # Pré-traitement de la translittération selon la langue source
        transliterated_source_term = source_candidate.lower()
        if source_language == "Greek" and greek_dict:
            for entry in greek_dict:
                if entry.get('greek') == source_candidate:  # Utiliser source_candidate (avec majuscule)
                    transliterated_source_term = entry.get('transliteration', source_candidate).lower()
                    break
            else:
                # Fallback uroman si pas trouvé
                if uroman_instance is None:
                    import uroman as ur
                    uroman_instance = ur.Uroman()
                transliterated_source_term = uroman_instance.romanize_string(source_candidate)


        elif source_language == "Hebrew":
            # Initialiser avec uroman par défaut
            if uroman_instance:
                transliterated_source_term = uroman_instance.romanize_string(source_candidate)
            if hebrew_dict:
                # Chercher la translittération la plus longue
                best_transliteration = None
                for entry in hebrew_dict:
                    if source_candidate in entry.get('hebrew_pure', []):
                        transliterations = entry.get('translitterations', [])
                        if transliterations:
                            current_trans = transliterations[0]
                            if best_transliteration is None or len(current_trans) > len(best_transliteration):
                                best_transliteration = current_trans
                
                if best_transliteration:
                    transliterated_source_term = best_transliteration
                    
        else:
            pass  # Pour les langues non-chinoises

            
        source_term_normalized = transliterated_source_term.lower()
                        
        # Filtrage par similarité pinyin
        
        for target_term, count in data['target_counts'].items():
            # Calculer la similarité pinyin si langue cible est chinoise
            if target_language == "Chinese":
                # Comparaison pinyin entre nom anglais et chinois
                target_pinyin = get_pinyin(target_term, pinyin_dictionary)
                
                # Similarité phonétique directe
                similarity_score = calculate_name_similarity(source_term_normalized, target_pinyin, source_language)
                # Bonus pour longueur appropriée des noms propres chinois
                if 2 <= len(target_term) <= 4:
                    similarity_score += 0.2
                
                # Calculer longueur cible attendue selon le nom source
                source_length = len(source_candidate)
                if source_length <= 5: # Noms courts
                    target_length_range = (2, 3)
                elif source_length <= 7: # Noms moyens
                    target_length_range = (3, 4)
                else:  # Noms longs
                    target_length_range = (3, 5)
                target_length = len(target_term)
                
                if not (target_length_range[0] <= target_length <= target_length_range[1]):
                    continue  # Passer au target_term suivant
                    
                # Caractères de liaisons en début de mot
                liaison_first_chars = {'子', '的', '他', '憑', '信', '為', '與', '從', '在', '和', '之', '了', '生', '於', '而'}
                if len(target_term) >= 2 and target_term[0] in liaison_first_chars:
                    continue
                
            else:
                similarity_score = 0.6  # Score par défaut pour non-chinois
                
            if similarity_score >= 0.56:
                candidate_associations.append({
                    'source_term': source_term_normalized,
                    'source_candidate': source_candidate,
                    'target_term': target_term,
                    'count': count,
                    'similarity': similarity_score
                })
    
    # Trier par similarité décroissante
    candidate_associations.sort(key=lambda x: x['similarity'], reverse=True)

    # Résoudre les conflits : chaque target_term ne peut être attribué qu'une fois
    used_targets = set()
    resolved_associations = []
    
    for assoc in candidate_associations:
        if assoc['target_term'] not in used_targets:
            resolved_associations.append(assoc)
            used_targets.add(assoc['target_term'])
    
    # ===== ÉTAPE 5: RÉATTRIBUTION DES PERDANTS =====
    
    # Identifier les termes source qui n'ont pas été attribués
    resolved_sources = {assoc['source_term'] for assoc in resolved_associations}
    all_sources = {candidate.lower() for candidate in filtered_candidates.keys()}
    losing_sources = all_sources - resolved_sources
    
    # Essayer de réattribuer avec leurs alternatives
    for losing_source in losing_sources:
        # Trouver le candidat original (avec majuscule)
        original_candidate = None
        for candidate in filtered_candidates.keys():
            if candidate.lower() == losing_source:
                original_candidate = candidate
                break
        
        if not original_candidate:
            continue
            
        # Chercher des alternatives non utilisées
        available_targets = []
        for target_term, count in filtered_candidates[original_candidate]['target_counts'].items():
            if target_term not in used_targets:
                # Calculer similarité pour l'alternative
                if target_language == "Chinese":
                    target_pinyin = get_pinyin(target_term, pinyin_dictionary)
                    similarity_score = calculate_name_similarity(losing_source, target_pinyin, source_language)
                    if 2 <= len(target_term) <= 4:
                        similarity_score += 0.2
                else:
                    similarity_score = 0.3
                
                available_targets.append({
                    'target_term': target_term,
                    'count': count,
                    'similarity': similarity_score
                })
        
        # Prendre la meilleure alternative si elle existe
        if available_targets:
            best_alternative = max(available_targets, key=lambda x: x['similarity'])
            
            # Vérifier un seuil minimum de qualité
            if best_alternative['similarity'] >= 0.5:
                resolved_associations.append({
                    'source_term': losing_source,
                    'source_candidate': original_candidate,
                    'target_term': best_alternative['target_term'],
                    'count': best_alternative['count'],
                    'similarity': best_alternative['similarity']
                })
                used_targets.add(best_alternative['target_term'])
                
    # Identifier les sources non résolues après réattribution
    final_resolved_sources = {assoc['source_candidate'].lower() for assoc in resolved_associations}
    truly_unresolved = all_sources - final_resolved_sources
    
    for unresolved_source in truly_unresolved:
        original_candidate = None
        for candidate in filtered_candidates.keys():
            if candidate.lower() == unresolved_source:
                original_candidate = candidate
                break
        unresolved_proper_nouns.append(original_candidate if original_candidate else unresolved_source)
    
    # ===== INTÉGRATION DANS LE DICTIONNAIRE APPRIS =====
    
    for assoc in resolved_associations:
        source_term = assoc['source_candidate']
        target_term = assoc['target_term']
        similarity_score = assoc['similarity']
        
        # Score pour les noms propres : 5
        proper_noun_score = similarity_score*6.0
        
        if source_term in learned_dictionary["source_term_entry"]:
            # Terme existant - mise à jour
            learned_targets = learned_dictionary["source_term_entry"][source_term]["learned_targets"]
            
            target_found = False
            for target in learned_targets:
                if target["target_term"] == target_term:
                    target["confidence_score"] += proper_noun_score
                    target["is_primary"] = True
                    target_found = True
                    break
            
            if not target_found:
                learned_targets.append({
                    "target_term": target_term,
                    "confidence_score": proper_noun_score,
                    "is_primary": True
                })
            
            learned_dictionary["source_term_entry"][source_term]["last_updated"] = batch_reference
            updated_terms_summary.append(f"{source_term}→{target_term}")
            action = "updated"
            
        else:
            # Nouveau terme
            learned_targets = [{
                "target_term": target_term,
                "confidence_score": proper_noun_score,
                "is_primary": True
            }]
            
            learned_dictionary["source_term_entry"][source_term] = {
                "learned_targets": learned_targets,
                "first_learned": batch_reference,
                "last_updated": batch_reference
            }
            
            learned_terms_summary.append(f"{source_term}→{target_term}")
            action = "learned"
        
        # Ajouter à l'historique
        learned_dictionary["cumulative_data"]["learning_history"].append({
            "batch_reference": batch_reference,
            "source_term": source_term,
            "action": action,
            "primary_target": target_term,
            "learning_method": "proper_noun",
            "similarity_score": similarity_score,
            "timestamp": datetime.now().isoformat()
        })
        
        retained_source_terms.add(source_term)
        retained_target_terms.add(target_term)
    
    return {
        'proper_noun_learned_terms': learned_terms_summary,
        'proper_noun_updated_terms': updated_terms_summary,
        'proper_noun_retained_terms': retained_source_terms,
        'proper_noun_unresolved': unresolved_proper_nouns
    }

def detect_hebrew_proper_nouns(text):
    """
    Détecte les noms propres en hébreu en consultant le fichier heb_eng.json
    Retourne un dictionnaire {terme_pure: nombre_occurrences} avec les formes non vocalisées
    Privilégie la forme hebrew_pure la plus courte (biblique) en cas de variantes multiples
    """
    try:
        import json
        from collections import Counter
        
        # Charger le dictionnaire hébreu-anglais
        with open('input/dictionaries/heb_eng.json', 'r', encoding='utf-8') as f:
            hebrew_dict = json.load(f)
        
        # Créer un mapping : forme_vocalisée -> forme_pure la plus courte
        vocalized_to_pure = {}
        
        for entry in hebrew_dict:
            if entry.get('proper_name', False):
                pure_forms = entry.get('hebrew_pure', [])
                vocalized_forms = entry.get('hebrew', [])
                
                for vocalized in vocalized_forms:
                    for pure in pure_forms:
                        # Si le terme vocalisé existe déjà, garder la forme pure la plus courte
                        if vocalized in vocalized_to_pure:
                            current_pure = vocalized_to_pure[vocalized]
                            if len(pure) < len(current_pure):
                                vocalized_to_pure[vocalized] = pure
                        else:
                            vocalized_to_pure[vocalized] = pure
        
        # Diviser le texte en mots hébreux
        import re
        words = re.findall(r'[\u0590-\u05FF]+', text)
        
        # Chercher les formes vocalisées dans le texte et convertir en formes pures
        proper_nouns_pure = []
        for word in words:
            if word in vocalized_to_pure:
                proper_nouns_pure.append(vocalized_to_pure[word])
        
        # Retourner les comptes des formes pures
        return dict(Counter(proper_nouns_pure))
        
    except Exception as e:
        print(f"Erreur lors de la détection des noms propres hébreux: {e}")
        return {}

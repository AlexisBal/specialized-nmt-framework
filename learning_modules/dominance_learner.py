"""
Dominance Learner
Apprends les termes dominants sur les cooccurrences cumulées d'un batch
"""
import json
from datetime import datetime
from collections import Counter, defaultdict


def learn_from_dominance_batch(
    cooccurrences_batch,        # Cooccurrences accumulées du batch
    batch_reference,            # Référence du batch
    learned_dictionary,         # Dictionnaire des termes appris
    source_language,            # Langue source
    target_language             # Langue cible
):
    """
    Apprend les termes par dominance sur un batch complet
    Critère: terme target 3x plus fréquent que les autres
    """
    
    learned_terms_summary = []
    updated_terms_summary = []
    retained_source_terms = set()
    retained_target_terms = set()

    for source_term, target_counts in cooccurrences_batch.items():
        available_targets = {
            target: count for target, count in target_counts.items()
            if target not in retained_target_terms
        }
        
        if len(available_targets) < 2:  # Pas de dominance possible avec un seul terme
            continue
            
        # Calculer la dominance
        sorted_targets = sorted(available_targets.items(), key=lambda x: x[1], reverse=True)
        max_count = sorted_targets[0][1]
        terms_at_max = sum(1 for target, count in sorted_targets if count == max_count) # groupe de termes dominants
        total_count = sum(available_targets.values())
        
        if terms_at_max <= 9 and max_count >= 4:
            
            # Identifier les termes dominants (en cas d'égalité au maximum)
            dominant_targets = [target for target, count in sorted_targets if count == max_count]
    
            # Filtrage des caractères génériques pour le chinois
            if target_language == "Chinese":
                # Caractères génériques (retour d'expérience)
                generic_chars = {'的', '了', '與', '和', '或', '但', '因', '從', '到', '向', '於', '為', '由', '及', '而', '我', '你', '他', '沒', '有', '人', '說', '一', '不', '是', '在', '來', '去', '出', '過', '會', '能', '要', '可', '就', '都', '也', '還', '又', '已', '裏', '內', '去'
                }
                
                # Filtrer les termes dominants avec caractères génériques
                filtered_dominant_targets = []
                for target in dominant_targets:
                    has_generic = (target[0] in generic_chars or target[-1] in generic_chars) if target else True
                    if not has_generic:
                        filtered_dominant_targets.append(target)
                
                # Continuer seulement si on a des termes valides après filtrage
                if not filtered_dominant_targets:
                    continue
                    
                dominant_targets = filtered_dominant_targets
            
            # Choisir le terme le plus long comme primaire
            primary_target = max(dominant_targets, key=len)
            alternative_targets = [t for t in dominant_targets if t != primary_target]
            
            # Score de dominance
            dominance_score = 5.0
            
            # Mise à jour du dictionnaire appris
            if source_term in learned_dictionary["source_term_entry"]:
                # Terme existant - mise à jour
                learned_targets = learned_dictionary["source_term_entry"][source_term]["learned_targets"]
                
                primary_found = False
                
                for target in learned_targets:
                    if target["target_term"] == primary_target:
                        target["confidence_score"] += dominance_score
                        target["is_primary"] = True
                        primary_found = True
                        break
                
                if not primary_found:
                    learned_targets.append({
                        "target_term": primary_target,
                        "confidence_score": dominance_score,
                        "is_primary": True
                    })
                
                # Ajouter les alternatives
                for alt_target in alternative_targets:
                    alt_found = False
                    for target in learned_targets:
                        if target["target_term"] == alt_target:
                            target["confidence_score"] += dominance_score * 0.8  # Score réduit pour alternatives
                            alt_found = True
                            break
                    if not alt_found:
                        learned_targets.append({
                            "target_term": alt_target,
                            "confidence_score": dominance_score * 0.8,
                            "is_primary": False
                        })
                
                learned_dictionary["source_term_entry"][source_term]["last_updated"] = batch_reference
                updated_terms_summary.append(f"{source_term}→{primary_target}")
                action = "updated"
                
            else:
                # Nouveau terme
                learned_targets = []
                
                # Ajouter le terme primaire
                learned_targets.append({
                    "target_term": primary_target,
                    "confidence_score": dominance_score,
                    "is_primary": True
                })
                
                # Ajouter les alternatives
                for alt_target in alternative_targets:
                    learned_targets.append({
                        "target_term": alt_target,
                        "confidence_score": dominance_score * 0.8,
                        "is_primary": False
                    })
                
                learned_dictionary["source_term_entry"][source_term] = {
                    "learned_targets": learned_targets,
                    "first_learned": batch_reference,
                    "last_updated": batch_reference
                }
                
                learned_terms_summary.append(f"{source_term}→{primary_target}")
                action = "learned"
                
            retained_source_terms.add(source_term)
            retained_target_terms.add(primary_target)
            retained_target_terms.update(alternative_targets)
            
            # Ajouter à l'historique
            learned_dictionary["cumulative_data"]["learning_history"].append({
                "batch_reference": batch_reference,
                "source_term": source_term,
                "action": action,
                "primary_target": primary_target,
                "alternatives_found": alternative_targets,
                "learning_method": "dominance",
                "dominance_ratio": max_count,
                "timestamp": datetime.now().isoformat()
            })
            
    # Filtrer les cooccurrences pour l'étape suivante

    cleaned_proper_cooccurrences = {
        source: {
            target: count for target, count in targets.items()
            if target not in retained_target_terms and
               not any(retained_term in target for retained_term in retained_target_terms)
        }
        for source, targets in cooccurrences_batch.items()
        if source not in retained_source_terms
    }
    
    return {
        'dominance_batch_learned_terms': learned_terms_summary,
        'dominance_batch_updated_terms': updated_terms_summary,
        'dominance_retained_target_terms': retained_target_terms,
        'dominance_retained_source_terms': retained_source_terms,
        'cleaned_proper_cooccurrences': cleaned_proper_cooccurrences
    }

"""
Learner Batch Summary
Génère des rapports de synthèse pour les résultats d'apprentissage par batch
"""

def print_batch_learning_summary(batch_reference, verse_count, total_dict_learned, total_dict_updated,
                                total_similarity_learned, total_similarity_updated, dominance_result,
                                proper_nouns_result):
    """
    Affiche un rapport de synthèse des apprentissages du batch
    
    Args:
        batch_reference: Référence du batch (ex: "1_1_1-4_13")
        verse_count: Nombre de versets traités
        total_dict_learned: Liste des termes appris par dictionnaire
        total_dict_updated: Liste des termes mis à jour par dictionnaire
        total_similarity_learned: Liste des termes appris par similarité
        total_similarity_updated: Liste des termes mis à jour par similarité
        dominance_result: Résultats de l'apprentissage par dominance
        proper_nouns_result: Résultats de l'apprentissage des noms propres
    """
    
    print(f"\n" + "="*80)
    print(f"📊 RAPPORT D'APPRENTISSAGE - BATCH {batch_reference}")
    print(f"="*80)
    
    # Statistiques générales
    print(f"📝 Versets traités: {verse_count}")
    
    # Détail par méthode d'apprentissage
    print(f"\n🔍 RÉSULTATS PAR MÉTHODE:")
    
    # 1. Dictionnaire
    dict_learned_count = len(total_dict_learned)
    dict_updated_count = len(total_dict_updated)
    print(f"   📚 Dictionnaire    : +{dict_learned_count:2d} appris  | +{dict_updated_count:2d} mis à jour")
    if dict_learned_count > 0:
        examples = " | ".join(total_dict_learned[:20])
        print(f"      Exemples       : {examples}")
    
    # 2. Similarité
    sim_learned_count = len(total_similarity_learned)
    sim_updated_count = len(total_similarity_updated)
    print(f"   🔗 Similarité      : +{sim_learned_count:2d} appris  | +{sim_updated_count:2d} mis à jour")
    if sim_learned_count > 0:
        examples = " | ".join(total_similarity_learned)
        print(f"      Exemples       : {examples}")
    
    # 3. Dominance
    dom_learned_count = len(dominance_result['dominance_batch_learned_terms'])
    dom_updated_count = len(dominance_result['dominance_batch_updated_terms'])
    print(f"   💪 Dominance       : +{dom_learned_count:2d} appris  | +{dom_updated_count:2d} mis à jour")
    if dom_learned_count > 0:
        examples = " | ".join(dominance_result['dominance_batch_learned_terms'])
        print(f"      Exemples       : {examples}")
    
    # 4. Noms propres
    prop_learned_count = len(proper_nouns_result['proper_noun_learned_terms'])
    prop_updated_count = len(proper_nouns_result['proper_noun_updated_terms'])
    prop_unresolved_count = len(proper_nouns_result.get('proper_noun_unresolved', []))
    print(f"   👤 Noms propres    : +{prop_learned_count:2d} appris  | +{prop_updated_count:2d} mis à jour | {prop_unresolved_count:2d} non résolus")
    if prop_learned_count > 0:
        examples = " | ".join(proper_nouns_result['proper_noun_learned_terms'])
        print(f"      Exemples       : {examples}")
    if prop_unresolved_count > 0:
        unresolved_list = proper_nouns_result.get('proper_noun_unresolved', [])
        if unresolved_list:
            nonresolus = " | ".join(str(item) for item in unresolved_list)
            print(f"      Non résolus       : {nonresolus}")
    
    # Totaux
    total_learned_all = (dict_learned_count + sim_learned_count + dom_learned_count +
                        prop_learned_count)
    total_updated_all = (dict_updated_count + sim_updated_count + dom_updated_count +
                        prop_updated_count)
    
    print(f"\n🎯 TOTAL BATCH:")
    print(f"   ✅ Nouveaux termes appris : {total_learned_all}")
    print(f"   🔄 Termes mis à jour      : {total_updated_all}")
    print(f"   📈 Impact total           : {total_learned_all + total_updated_all}")
    
    # Efficacité par verset
    if verse_count > 0:
        avg_learned_per_verse = total_learned_all / verse_count
        print(f"   📊 Moyenne par verset     : {avg_learned_per_verse:.1f} termes/verset")
    
    print(f"="*80)


def get_batch_learning_totals(total_dict_learned, total_dict_updated, total_similarity_learned,
                             total_similarity_updated, dominance_result, proper_nouns_result):
    """
    Calcule les totaux consolidés d'apprentissage pour un batch
    
    Returns:
        tuple: (total_learned, total_updated) incluant tous les types d'apprentissage
    """
    
    total_learned = (
        len(total_dict_learned) +
        len(total_similarity_learned) +
        len(dominance_result['dominance_batch_learned_terms']) +
        len(proper_nouns_result['proper_noun_learned_terms'])
    )
    
    total_updated = (
        len(total_dict_updated) +
        len(total_similarity_updated) +
        len(dominance_result['dominance_batch_updated_terms']) +
        len(proper_nouns_result['proper_noun_updated_terms'])
    )
    
    return total_learned, total_updated


def print_compact_batch_summary(batch_reference, verse_count, total_learned, total_updated):
    """
    Affiche un résumé compact du batch (pour logs rapides)
    """
    
    avg_per_verse = total_learned / verse_count if verse_count > 0 else 0
    print(f"📊 Batch {batch_reference}: {verse_count} versets | +{total_learned} appris | +{total_updated} maj | {avg_per_verse:.1f}/verset")

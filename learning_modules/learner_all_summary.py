"""
Learner All Summary
Génère des rapports de synthèse globaux pour tous les batches d'apprentissage
"""

import numpy as np
import matplotlib.pyplot as plt

# Stockage global des données d'apprentissage pour le rapport final
_ALL_LEARNING_DATA = []

def store_batch_learning_data(batch_reference, verse_count, total_dict_learned,
                             total_similarity_learned, dominance_result, proper_nouns_result):
    """
    Stocke les données d'apprentissage d'un batch pour le rapport final
    
    Args:
        batch_reference: Référence du batch
        verse_count: Nombre de versets traités
        total_dict_learned: Liste des termes appris par dictionnaire
        total_similarity_learned: Liste des termes appris par similarité
        dominance_result: Résultats de l'apprentissage par dominance
        proper_nouns_result: Résultats de l'apprentissage des noms propres
    """
    global _ALL_LEARNING_DATA
    
    dict_learned_count = len(total_dict_learned)
    sim_learned_count = len(total_similarity_learned)
    dom_learned_count = len(dominance_result['dominance_batch_learned_terms'])
    prop_learned_count = len(proper_nouns_result['proper_noun_learned_terms'])
    prop_unresolved_count = len(proper_nouns_result.get('proper_noun_unresolved', []))
    
    total_learned = dict_learned_count + sim_learned_count + dom_learned_count + prop_learned_count
    
    batch_data = {
        'batch_reference': batch_reference,
        'verse_count': verse_count,
        'dict_learned': dict_learned_count,
        'similarity_learned': sim_learned_count,
        'dominance_learned': dom_learned_count,
        'proper_noun_learned': prop_learned_count,
        'proper_noun_unresolved': prop_unresolved_count,
        'total_learned': total_learned,
        'avg_learned_per_verse': total_learned / verse_count if verse_count > 0 else 0
    }
    
    _ALL_LEARNING_DATA.append(batch_data)

def print_global_learning_summary():
    """
    Affiche un rapport de synthèse global pour tous les batches d'apprentissage
    """
    global _ALL_LEARNING_DATA
    
    if not _ALL_LEARNING_DATA:
        print("⚠️ Aucune donnée d'apprentissage disponible")
        return
    
    print(f"\n" + "="*100)
    print(f"📚 RAPPORT GLOBAL D'APPRENTISSAGE - {len(_ALL_LEARNING_DATA)} BATCHES")
    print(f"="*100)
    
    # Calculs des totaux globaux
    _print_global_totals()
    
    # Tableau détaillé par batch
    _print_learning_table()
    
    # Analyse des proportions
    _print_learning_proportions()
    
    # Courbes d'évolution
    _print_learning_evolution()

def _print_global_totals():
    """
    Affiche les totaux globaux consolidés
    """
    total_verses = sum(m['verse_count'] for m in _ALL_LEARNING_DATA)
    total_learned_global = sum(m['total_learned'] for m in _ALL_LEARNING_DATA)
    
    # Totaux par catégorie
    total_dict_learned = sum(m['dict_learned'] for m in _ALL_LEARNING_DATA)
    total_sim_learned = sum(m['similarity_learned'] for m in _ALL_LEARNING_DATA)
    total_dom_learned = sum(m['dominance_learned'] for m in _ALL_LEARNING_DATA)
    total_prop_learned = sum(m['proper_noun_learned'] for m in _ALL_LEARNING_DATA)
    total_unresolved = sum(m['proper_noun_unresolved'] for m in _ALL_LEARNING_DATA)
    
    print(f"\n🎯 TOTAUX GLOBAUX:")
    print(f"   📖 Total versets traités        : {total_verses:,}")
    print(f"   ✅ Total nouveaux termes appris : {total_learned_global:,}")
    print(f"   📊 Moyenne par verset           : {total_learned_global / total_verses:.2f} termes/verset")
    

def _print_learning_proportions():
    """
    Calcule et affiche les proportions par catégorie
    """
    total_learned_global = sum(m['total_learned'] for m in _ALL_LEARNING_DATA)
    
    if total_learned_global == 0:
        print("\n⚠️ Aucun terme appris pour calculer les proportions")
        return
    
    total_dict_learned = sum(m['dict_learned'] for m in _ALL_LEARNING_DATA)
    total_sim_learned = sum(m['similarity_learned'] for m in _ALL_LEARNING_DATA)
    total_dom_learned = sum(m['dominance_learned'] for m in _ALL_LEARNING_DATA)
    total_prop_learned = sum(m['proper_noun_learned'] for m in _ALL_LEARNING_DATA)
    
    print(f"   📚 Dictionnaire     : {total_dict_learned / total_learned_global * 100:5.1f}% ({total_dict_learned:,} termes)")
    print(f"   💪 Dominance        : {total_dom_learned / total_learned_global * 100:5.1f}% ({total_dom_learned:,} termes)")
    print(f"   🔗 Similarité       : {total_sim_learned / total_learned_global * 100:5.1f}% ({total_sim_learned:,} termes)")
    print(f"   👤 Noms propres     : {total_prop_learned / total_learned_global * 100:5.1f}% ({total_prop_learned:,} termes)")
    

def _print_learning_table():
    """
    Tableau détaillé avec données en lignes, batches en colonnes
    """
    batch_references = [m['batch_reference'] for m in _ALL_LEARNING_DATA]
    
    print(f"\n📋 DÉTAIL PAR BATCH:")
    
    # En-tête du tableau
    header = f"{'Batche':<20}"
    for i, batch_ref in enumerate(batch_references, 1):
        header += f"{i:<10}"
    
    print(f"\n{header}")
    print("-" * len(header))
    
    # Lignes du tableau
    data_to_show = [
        ('Versets', 'verse_count', 'd'),
        ('Total appris', 'total_learned', 'd'),
        ('Dictionnaire', 'dict_learned', 'd'),
        ('Dominance', 'dominance_learned', 'd'),
        ('Similarité', 'similarity_learned', 'd'),
        ('Noms propres', 'proper_noun_learned', 'd'),
        ('Moy/verset', 'avg_learned_per_verse', '.1f')
    ]
    
    for metric_name, data_key, format_str in data_to_show:
        line = f"{metric_name:<20}"
        
        total_value = 0
        for batch in _ALL_LEARNING_DATA:
            value = batch[data_key]
            if format_str == 'd':
                line += f"{value:<10d}"
            else:  # format_str == '.1f'
                line += f"{value:<10.1f}"
            if data_key != 'avg_learned_per_verse':  # Pas de somme pour les moyennes
                total_value += value
        
        # Total (sauf pour les moyennes)
        if data_key == 'avg_learned_per_verse':
            avg_global = sum(m['total_learned'] for m in _ALL_LEARNING_DATA) / sum(m['verse_count'] for m in _ALL_LEARNING_DATA)
            line += f"{avg_global:<10.1f}"
        else:
            if format_str == 'd':
                line += f"{total_value:<10d}"
            else:
                line += f"{total_value:<10.1f}"
        
        print(line)
    
    print("-" * len(header))

def _print_learning_evolution():
    """
    Affiche l'évolution cumulative des apprentissages par batch avec couleurs par méthode
    """
    print(f"\n📈 ÉVOLUTION CUMULATIVE DE L'APPRENTISSAGE:")
    print("-" * 60)
    
    # Calculer les totaux cumulés pour chaque méthode
    cumulative_dict = 0
    cumulative_dom = 0
    cumulative_sim = 0
    cumulative_prop = 0
    
    batches = []
    cumul_dict_values = []
    cumul_dom_values = []
    cumul_sim_values = []
    cumul_prop_values = []
    cumul_total_values = []
    
    for m in _ALL_LEARNING_DATA:
        cumulative_dict += m['dict_learned']
        cumulative_dom += m['dominance_learned']
        cumulative_sim += m['similarity_learned']
        cumulative_prop += m['proper_noun_learned']
        
        batches.append(m['batch_reference'])
        cumul_dict_values.append(cumulative_dict)
        cumul_dom_values.append(cumulative_dom)
        cumul_sim_values.append(cumulative_sim)
        cumul_prop_values.append(cumulative_prop)
        cumul_total_values.append(cumulative_dict + cumulative_dom + cumulative_sim + cumulative_prop)
    
    # Afficher la courbe cumulative avec couleurs empilées
    print(f"\n📚 Total cumulé par méthode:")
    _print_stacked_chart(batches, cumul_dict_values, cumul_dom_values, cumul_sim_values, cumul_prop_values, cumul_total_values)

def _print_cumulative_chart(batches, values, data_name):
    """Affiche un graphique cumulatif"""
    max_val = max(values)
    print(f"   {data_name} | Max: {max_val}")
    print("   " + "-" * 50)
    
    for batch, value in zip(batches, values):
        short_batch = batch.split('_')[-1] if '_' in batch else batch
        normalized = int((value / max_val) * 120) if max_val > 0 else 0
        bar = "▐" + "█" * normalized + "░" * (120 - normalized) + "▌"
        print(f"   {str(short_batch):<8} {bar} {value}")

def get_all_learning_data():
    """
    Retourne toutes les données d'apprentissage stockées
    """
    return _ALL_LEARNING_DATA.copy()

def clear_learning_data():
    """
    Nettoie les données d'apprentissage stockées (pour un nouveau run)
    """
    global _ALL_LEARNING_DATA
    _ALL_LEARNING_DATA = []
    
def _print_stacked_chart(batches, dict_values, dom_values, sim_values, prop_values, total_values):
    """Affiche un graphique empilé avec couleurs"""
    
    # Codes couleurs ANSI
    BLUE = '\033[94m'   # Dictionnaire
    RED = '\033[91m'    # Dominance
    GREEN = '\033[92m'  # Similarité
    YELLOW = '\033[93m' # Noms propres
    RESET = '\033[0m'
    
    max_val = max(total_values)
    print(f"   Total cumulé | Max: {max_val}")
    print("   " + "-" * 50)
    
    for i, (batch, total) in enumerate(zip(batches, total_values)):
        short_batch = str(i + 1)
        
        # Calculer les vrais totaux cumulés pour chaque méthode
        dict_cumul = dict_values[i]
        dom_cumul = dict_values[i] + dom_values[i]
        sim_cumul = dict_values[i] + dom_values[i] + sim_values[i]
        prop_cumul = total  # Le total final
        
        # Calculer les tailles des segments
        dict_size = int((dict_cumul / max_val) * 120) if max_val > 0 else 0
        dom_size = int((dom_cumul / max_val) * 120) - dict_size if max_val > 0 else 0
        sim_size = int((sim_cumul / max_val) * 120) - dict_size - dom_size if max_val > 0 else 0
        prop_size = int((prop_cumul / max_val) * 120) - dict_size - dom_size - sim_size if max_val > 0 else 0
        
        empty = 120 - (dict_size + dom_size + sim_size + prop_size)
        
        # Construire la barre colorée
        bar = "▐"
        bar += BLUE + "█" * dict_size + RESET
        bar += RED + "█" * dom_size + RESET
        bar += GREEN + "█" * sim_size + RESET
        bar += YELLOW + "█" * prop_size + RESET
        bar += "░" * empty + "▌"
        
        print(f"   {short_batch:<8} {bar} {total}")
    
    # Légende
    print(f"\n   Légende: {BLUE}█{RESET}Dictionnaire {RED}█{RESET}Dominance {GREEN}█{RESET}Similarité {YELLOW}█{RESET}Noms propres")

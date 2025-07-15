"""
Tuner All Summary
Génère des rapports de synthèse globaux pour tous les batches de fine-tuning
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

def print_global_tuning_summary():
    """
    Affiche un tableau des métriques (lignes) par batch (colonnes) + courbes
    """
    from tuner.tuner_batch_summary import get_all_batch_metrics
    
    all_metrics = get_all_batch_metrics()
    
    # BASELINE TEMPORAIRE - À SUPPRIMER AU PROCHAIN RUN
    if all_metrics and not any(m.get('batch_number') == 0 for m in all_metrics):
        print("🔧 Ajout baseline temporaire pour ce run...")
        temp_baseline = {
            'batch_number': 0,
            'bleu_score': 0.018,
            'rouge_score': 0.137,
            'comet_score': -0.094,  # Valeur négative = pas d'affichage
            'meteor_score': 0,
            'loss_improvement': 0,
            'training_time': 0,
            'verses_evaluated': 84
        }
        all_metrics.insert(0, temp_baseline)  # Insérer en premier
    
    # Sauvegarde de sécurité des métriques
    output_dir = "output/models"
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "all_batches_metrics.json")
    
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        print(f"✅ Métriques sauvegardées dans: {json_path}")
    except Exception as e:
        print(f"⚠️ Erreur lors de la sauvegarde JSON: {e}")
    
    if not all_metrics:
        print("⚠️ Aucune métrique de tuning disponible")
        return
    
    print(f"\n" + "="*100)
    print(f"📊 RAPPORT GLOBAL DE FINE-TUNING - {len(all_metrics)} BATCHES")
    print(f"="*100)
    
    # Construire le tableau
    _print_metrics_table(all_metrics)
    
    # Analyse complète (tous les batches)
    _analyze_metrics(all_metrics, "")
    
    # Analyse sans batch 1 (si plus d'un batch)
    if len(all_metrics) > 1:
        filtered_metrics = [m for m in all_metrics if m['batch_number'] != 1]
        _analyze_metrics(filtered_metrics, " (SANS BATCH 1)")
        
        # Comparaison avec/sans batch 1
        _print_comparison_summary(all_metrics, filtered_metrics)

def _print_metrics_table(all_metrics):
    """
    Tableau avec métriques en lignes, batches en colonnes
    """
    # Séparer baseline et batches
    baseline_data = None
    batch_data = []

    for metric in all_metrics:
        if metric['batch_number'] == 0:
            baseline_data = metric
        else:
            batch_data.append(metric)

    # En-tête du tableau
    header = f"{'Métrique':<20}"

    if baseline_data:
        header += f"{'BASE':<10}"

    for batch in batch_data:
        header += f"{str(batch['batch_number']):<10}"
    header += f"{'Moyenne':<10}"
    
    print(f"\n{header}")
    print("-" * len(header))
    
    # Lignes du tableau
    metrics_to_show = [
        ('BLEU', 'bleu_score'),
        ('ROUGE-L', 'rouge_score'),
        ('COMET', 'comet_score'),
        ('METEOR', 'meteor_score'),
        ('Loss Δ', 'loss_improvement'),
        ('Temps (s)', 'training_time'),
        ('Versets', 'verses_evaluated')
    ]
    
    for metric_name, metric_key in metrics_to_show:
        line = f"{metric_name:<20}"
        
        values = []
        
        # Baseline (sauf pour loss_improvement et training_time)
        if baseline_data:
            if metric_key in ['loss_improvement', 'training_time']:
                line += f"{'N/A':<10}"
            else:
                value = baseline_data[metric_key]
                if metric_key in ['comet_score', 'meteor_score'] and value == 0:
                    line += f"{'N/A':<10}"
                elif metric_key in ['verses_evaluated']:
                    line += f"{int(value):<10d}"
                else:
                    line += f"{value:<10.3f}"
        
        # Batches normaux
        for batch in batch_data:
            value = batch[metric_key]
            if metric_key == 'comet_score' and value == 0:
                line += f"{'N/A':<10}"
            elif metric_key == 'meteor_score' and value == 0:
                line += f"{'N/A':<10}"
            else:
                # Formatage selon le type de métrique
                if metric_key in ['training_time', 'verses_evaluated']:  # Pour les entiers
                    line += f"{int(value):<10d}"
                else:  # Pour les flottants
                    line += f"{value:<10.3f}"
                values.append(value)
    
        # Moyenne (seulement des batches, pas baseline)
        if values:
            avg = np.mean(values)
            # Même logique pour la moyenne
            if metric_key in ['training_time', 'verses_evaluated']:
                line += f"{int(avg):<10d}"
            else:
                line += f"{avg:<10.3f}"
        else:
            line += f"{'N/A':<10}"
        
        print(line)
    
    print("="*100)

def _analyze_metrics(all_metrics, title_suffix):
    """
    Fonction générique d'analyse des métriques (graphiques + tendances)
    """
    if not all_metrics:
        print(f"\n⚠️ Aucune donnée disponible{title_suffix}")
        return
    
    batches = [m['batch_number'] for m in all_metrics]
    
    # Séparer baseline et batches normaux
    baseline_data = None
    batch_data = []

    for metric in all_metrics:
        if metric['batch_number'] == 0:  # Baseline
            baseline_data = metric
        else:
            batch_data.append(metric)

    batches = [m['batch_number'] for m in batch_data]
    
    print(f"\n📈 GÉNÉRATION DES COURBES{title_suffix.upper()}...")
    if title_suffix:
        print(f"🚫 Analyse sur {len(all_metrics)} batches: {min(batches)} à {max(batches)}")
    
    # Générer les graphiques
    _plot_metrics(all_metrics, title_suffix, base_data)
    
    # Analyser les tendances
    _print_trends_analysis(all_metrics, title_suffix)

def _plot_metrics(all_metrics, title_suffix, baseline_data=None):
    """
    Génère les courbes d'évolution des métriques
    """
    if baseline_data is None:
        # Séparer baseline et batches normaux
        baseline_data = None
        batch_data = []
        for metric in all_metrics:
            if metric['batch_number'] == 0:
                baseline_data = metric
            else:
                batch_data.append(metric)
    else:
        batch_data = [m for m in all_metrics if m['batch_number'] != 0]

    batches = [m['batch_number'] for m in batch_data]
    # Métriques à afficher
    metrics_to_plot = [
        ('Loss Improvement', 'loss_improvement', 'red', '📉'),
        ('BLEU Score', 'bleu_score', 'blue', '🔤'),
        ('ROUGE-L Score', 'rouge_score', 'green', '📝'),
        ('COMET Score', 'comet_score', 'orange', '🌟')
    ]
    
    # Titre principal
    main_title = 'Évolution des Métriques de Fine-Tuning par Batch'
    if title_suffix:
        main_title += title_suffix.upper()
    
    # Créer une figure avec 4 sous-graphiques
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(main_title, fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for i, (metric_name, metric_key, color, emoji) in enumerate(metrics_to_plot):
        values = [m[metric_key] for m in batch_data]
        
        # Skip COMET si pas de valeurs valides
        if metric_key == 'comet_score' and all(v == 0 for v in values):
            axes[i].text(0.5, 0.5, 'Pas de données COMET valides',
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[i].transAxes, fontsize=12)
            axes[i].set_title(f'{emoji} {metric_name}')
            continue
        
        # Filtrer les valeurs nulles pour COMET
        if metric_key == 'comet_score':
            batch_values = [m[metric_key] for m in batch_data]
            valid_batch_data = [(m['batch_number'], m[metric_key]) for m in batch_data if m[metric_key] > 0]
            if valid_batch_data:
                plot_batches, plot_values = zip(*valid_batch_data)
            else:
                plot_batches, plot_values = [], []
        else:
            plot_batches, plot_values = batches, values
        
        if plot_values:
            # Ajouter la baseline si disponible (sauf pour loss_improvement)
            if baseline_data and metric_key != 'loss_improvement':
                baseline_value = baseline_data[metric_key]
                if metric_key != 'comet_score' or baseline_value > 0:
                    # Ligne horizontale pour la baseline
                    axes[i].axhline(y=baseline_value, color='red', linestyle=':',
                                   linewidth=2, alpha=0.7, label='Baseline (Modèle de Base)')
                    
                    # Point de baseline à x=0
                    axes[i].scatter([0], [baseline_value], color='red', s=80,
                                  marker='s', alpha=0.8, zorder=5)
            # Tracer la courbe principale
            axes[i].plot(plot_batches, plot_values, color=color, linewidth=2.5,
                        marker='o', markersize=5, alpha=0.8, label=metric_name)
            
            # Ajouter une ligne de tendance
            if len(plot_batches) > 1:
                z = np.polyfit(plot_batches, plot_values, 1)
                p = np.poly1d(z)
                axes[i].plot(plot_batches, p(plot_batches), color=color,
                            linestyle='--', alpha=0.6, linewidth=2, label='Tendance')
            
            # Statistiques
            min_val = min(plot_values)
            max_val = max(plot_values)
            avg_val = np.mean(plot_values)
            std_val = np.std(plot_values)
            
            # Titre avec statistiques
            suffix_display = title_suffix if title_suffix else ""
            axes[i].set_title(f'{emoji} {metric_name}{suffix_display}\n'
                             f'Min: {min_val:.3f} | Max: {max_val:.3f} | '
                             f'Moy: {avg_val:.3f} | Std: {std_val:.3f}')
            axes[i].set_xlabel('Numéro de Batch')
            axes[i].set_ylabel(metric_name)
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
            
            # Ajuster les limites pour une meilleure lisibilité
            if len(plot_values) > 1 and max_val > min_val:
                y_range = max_val - min_val
                y_margin = y_range * 0.1
                axes[i].set_ylim(min_val - y_margin, max_val + y_margin)
        else:
            axes[i].text(0.5, 0.5, 'Aucune donnée disponible',
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[i].transAxes, fontsize=12)
            axes[i].set_title(f'{emoji} {metric_name}')
    
    plt.tight_layout()
    
    # Nom de fichier selon le contexte
    if title_suffix:
        filename = 'fine_tuning_metrics_evolution_no_batch1.png'
    else:
        filename = 'fine_tuning_metrics_evolution.png'
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Graphiques sauvegardés dans: {filename}")
    
    # Afficher le graphique
    plt.show()

def _print_trends_analysis(all_metrics, title_suffix):
    """
    Affiche l'analyse textuelle des tendances
    """
    print(f"\n📊 RÉSUMÉ DES TENDANCES{title_suffix.upper()}:")
    print("-" * 60)
    
    metrics_info = [
        ('BLEU', 'bleu_score', '🔤'),
        ('ROUGE-L', 'rouge_score', '📝'),
        ('COMET', 'comet_score', '🌟'),
        ('Loss Δ', 'loss_improvement', '📉')
    ]
    
    for metric_name, metric_key, emoji in metrics_info:
        values = [m[metric_key] for m in all_metrics]
        
        if metric_key == 'comet_score':
            valid_values = [v for v in values if v > 0]
            if len(valid_values) < 2:
                print(f"{emoji} {metric_name:<15}: Données insuffisantes")
                continue
            values = valid_values
        
        if len(values) > 1:
            # Calculer la tendance
            batches_range = list(range(len(values)))
            slope = np.polyfit(batches_range, values, 1)[0]
            
            # Statistiques
            min_val = min(values)
            max_val = max(values)
            avg_val = np.mean(values)
            std_val = np.std(values)
            
            # Direction de la tendance
            if slope > 0.001:
                trend = "🔺 CROISSANTE"
            elif slope < -0.001:
                trend = "🔻 DÉCROISSANTE"
            else:
                trend = "➡️ STABLE"
            
            # Coefficient de variation (volatilité)
            cv = (std_val / avg_val) * 100 if avg_val != 0 else 0
            
            print(f"{emoji} {metric_name:<15}: {trend:<15} | "
                  f"Range: {min_val:.3f}-{max_val:.3f} | "
                  f"Moy: {avg_val:.3f} | "
                  f"Volatilité: {cv:.1f}%")
    
    print("-" * 60)

def _print_comparison_summary(all_metrics, filtered_metrics):
    """
    Affiche une comparaison des métriques avec et sans le batch 1
    """
    print("\n" + "="*80)
    print("COMPARAISON: IMPACT DU BATCH 1 SUR LES MOYENNES")
    print("="*80)
    
    # Fonction helper pour calculer les stats
    def calc_stats(metrics_list):
        if not metrics_list:
            return {'bleu': 0, 'rouge': 0, 'comet': 0, 'loss': 0}
        
        comet_values = [m['comet_score'] for m in metrics_list if m['comet_score'] > 0]
        return {
            'bleu': np.mean([m['bleu_score'] for m in metrics_list]),
            'rouge': np.mean([m['rouge_score'] for m in metrics_list]),
            'comet': np.mean(comet_values) if comet_values else 0,
            'loss': np.mean([m['loss_improvement'] for m in metrics_list])
        }
    
    stats_with_batch1 = calc_stats(all_metrics)
    stats_without_batch1 = calc_stats(filtered_metrics)
    
    print("MOYENNES:")
    print(f"{'Métrique':<15} {'Avec Batch 1':<15} {'Sans Batch 1':<15} {'Différence':<15}")
    print("-" * 60)
    
    for metric in ['bleu', 'rouge', 'comet', 'loss']:
        with_b1 = stats_with_batch1[metric]
        without_b1 = stats_without_batch1[metric]
        diff = with_b1 - without_b1
        print(f"{metric.upper():<15} {with_b1:<15.3f} {without_b1:<15.3f} {diff:<15.3f}")
    
    # Temps total
    total_time_all = sum([m['training_time'] for m in all_metrics])
    total_time_filtered = sum([m['training_time'] for m in filtered_metrics])
    
    print(f"\n⏱️ TEMPS D'ENTRAÎNEMENT:")
    print(f"   • Avec Batch 1    : {total_time_all:.0f}s ({total_time_all/3600:.1f}h)")
    print(f"   • Sans Batch 1    : {total_time_filtered:.0f}s ({total_time_filtered/3600:.1f}h)")
    print(f"   • Temps Batch 1   : {total_time_all - total_time_filtered:.0f}s")
    
    print("="*80)

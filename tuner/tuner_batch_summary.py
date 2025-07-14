"""
Tuner Batch Summary
Génère des rapports de synthèse pour les résultats du fine-tuning par batch
"""

# Stockage global des métriques pour le rapport final
_ALL_BATCH_METRICS = []
_BATCH_1_BASELINE = None

def print_batch_tuning_summary(batch_number, training_metrics, source_language, target_language):
    """
    Affiche un rapport de synthèse des métriques de fine-tuning du batch
    
    Args:
        batch_number: Numéro du batch
        training_metrics: Dictionnaire des métriques calculées par metrics_calculator
        source_language: Langue source
        target_language: Langue cible
    """
    
    print(f"\n" + "="*80)
    print(f"🎯 RAPPORT DE FINE-TUNING - BATCH {batch_number}")
    print(f"="*80)
    
    # Informations générales
    print(f"🌐 Direction: {source_language} → {target_language}")
    
    # Métriques de loss
    loss_metrics = training_metrics.get('loss_metrics', {})
    print(f"\n📉 MÉTRIQUES DE LOSS:")
    print(f"   📊 Loss final (train)    : {loss_metrics.get('final_train_loss', 0.0):.4f}")
    print(f"   📊 Loss final (eval)     : {loss_metrics.get('final_eval_loss', 0.0):.4f}")
    print(f"   ⭐ Meilleur loss (eval)  : {loss_metrics.get('best_eval_loss', 0.0):.4f}")
    print(f"   📈 Amélioration totale   : {loss_metrics.get('loss_improvement', 0.0):.4f}")
    
    # Métriques de convergence
    convergence_metrics = training_metrics.get('convergence_metrics', {})
    print(f"\n🎯 CONVERGENCE:")
    print(f"   🔄 Époques complétées   : {convergence_metrics.get('epochs_completed', 0.0)}")
    print(f"   ⏹️  Early stopping      : {'Oui' if convergence_metrics.get('early_stopped', False) else 'Non'}")
    print(f"   🏆 Époque de convergence : {convergence_metrics.get('convergence_epoch', 0.0)}")
    print(f"   📊 Steps d'entraînement : {convergence_metrics.get('total_training_steps', 0.0)}")
    
    # Métriques de performance
    performance_metrics = training_metrics.get('performance_metrics', {})
    print(f"\n⚡ PERFORMANCE:")
    print(f"   ⏱️  Temps d'entraînement : {performance_metrics.get('training_time_seconds', 0.0)} sec")
    print(f"   🚀 Tokens/seconde       : {performance_metrics.get('tokens_per_second', 0.0)}")
    print(f"   🔢 Tokens traités       : {performance_metrics.get('total_tokens_processed', 0.0)}")
    
    print(f"="*80)


def print_compact_tuning_summary(batch_number, training_metrics):
    """
    Affiche un résumé compact du fine-tuning (pour logs rapides)
    """
    
    bleu_score = training_metrics.get('posttuned_quality_metrics', {}).get('bleu_score', 0)
    loss_improvement = training_metrics.get('loss_metrics', {}).get('loss_improvement', 0)
    training_time = training_metrics.get('performance_metrics', {}).get('training_time_seconds', 0)
    comet_score = training_metrics.get('quality_metrics', {}).get('comet_score', 0)
    comet_str = f" | COMET={comet_score:.3f}" if comet_score else ""

    print(f"🎯 Tuning Batch {batch_number}: BLEU={bleu_score:.3f}{comet_str} | Loss Δ={loss_improvement:.3f} | {training_time:.0f}s")
    

def get_tuning_quality_score(training_metrics):
    """
    Calcule un score de qualité global du fine-tuning (0-100)
    
    Returns:
        int: Score de qualité entre 0 et 100
    """
    
    # Pondération des différentes métriques
    bleu_score = training_metrics.get('posttuned_quality_metrics', {}).get('bleu_score', 0)
    rouge_score = training_metrics.get('posttuned_quality_metrics', {}).get('rouge_l_score', 0)
    comet_score = training_metrics.get('posttuned_quality_metrics', {}).get('comet_score', 0)
    loss_improvement = training_metrics.get('loss_metrics', {}).get('loss_improvement', 0)

    # Normalisation et pondération
    bleu_contribution = min(bleu_score * 100, 40)  # Max 40 points
    rouge_contribution = min(rouge_score * 100, 25)  # Max 25 points
    comet_contribution = min(comet_score * 100, 25) if comet_score else 0  # Max 25 points
    loss_contribution = min(loss_improvement * 50, 10)  # Max 10 points

    total_score = int(bleu_contribution + rouge_contribution + comet_contribution + loss_contribution)
    
    return min(total_score, 100)

def store_batch_metrics(batch_number, training_metrics, source_language, target_language):
    """
    Stocke les métriques d'un batch pour le rapport final
    """
    global _ALL_BATCH_METRICS
    
    quality_metrics = training_metrics.get('posttuned_quality_metrics', {})
    loss_metrics = training_metrics.get('loss_metrics', {})
    performance_metrics = training_metrics.get('performance_metrics', {})
    
    batch_data = {
        'batch_number': batch_number,
        'source_language': source_language,
        'target_language': target_language,
        'bleu_score': quality_metrics.get('bleu_score', 0),
        'rouge_score': quality_metrics.get('rouge_l_score', 0),
        'comet_score': quality_metrics.get('comet_score', 0),
        'meteor_score': quality_metrics.get('meteor_score', 0),
        'loss_improvement': loss_metrics.get('loss_improvement', 0),
        'training_time': performance_metrics.get('training_time_seconds', 0),
        'verses_evaluated': quality_metrics.get('verses_evaluated', 0)
    }
    
    _ALL_BATCH_METRICS.append(batch_data)

def get_all_batch_metrics():
    """
    Retourne toutes les métriques stockées
    """
    result = []
    if _BATCH_1_BASELINE is not None:
        result.append(_BATCH_1_BASELINE.copy())
    result.extend(_ALL_BATCH_METRICS.copy())
    return result

def clear_batch_metrics():
    """
    Nettoie les métriques stockées (pour un nouveau run)
    """
    global _ALL_BATCH_METRICS, _BATCH_1_BASELINE
    _ALL_BATCH_METRICS = []
    _BATCH_1_BASELINE = None

def store_batch_1_baseline(training_metrics):
    """
    Stocke les métriques pretuned du batch 1 comme baseline de référence
    """
    global _BATCH_1_BASELINE
    
    pretuned_metrics = training_metrics.get('pretuned_quality_metrics', {})
    
    _BATCH_1_BASELINE = {
        'batch_number': 0,
        'bleu_score': pretuned_metrics.get('bleu_score', 0),
        'rouge_score': pretuned_metrics.get('rouge_l_score', 0),
        'comet_score': pretuned_metrics.get('comet_score', 0),
        'meteor_score': pretuned_metrics.get('meteor_score', 0),
        'loss_improvement': 0,
        'training_time': 0,
        'verses_evaluated': pretuned_metrics.get('verses_evaluated', 0)
    }

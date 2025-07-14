"""
Calculer les principales métriques pour un training donné - Version améliorée pour le chinois
Réorganisé en deux fonctions distinctes : métriques de training et métriques de traduction
"""

import torch, re
import gc
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import jieba
import random

from tuner.translator import get_mbart_language_code
from tuner.training_utils import preserve_chinese_punctuation

# Imports conditionnels pour les bibliothèques chinoises
try:
    from rouge_chinese import Rouge as RougeChineseLib
    ROUGE_CHINESE_AVAILABLE = True
except ImportError:
    ROUGE_CHINESE_AVAILABLE = False

# Import standard rouge_scorer
from rouge_score import rouge_scorer

try:
    import sacrebleu
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False
    
# Import conditionnel pour COMET
try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    
# Chargement global du modèle COMET
_COMET_MODEL = None

def _get_comet_model():
    """Charge COMET de façon simple"""
    global _COMET_MODEL
    if _COMET_MODEL is None and COMET_AVAILABLE:
        try:
            print("📊 Chargement COMET")
            model_path = download_model("Unbabel/eamt22-cometinho-da")
            _COMET_MODEL = load_from_checkpoint(model_path)
            print("✅ COMET chargé avec succès")
            
        except Exception as e:
            print(f"❌ Erreur COMET: {e}")
            _COMET_MODEL = False
            
    return _COMET_MODEL

def calculate_training_metrics(train_history, training_time, source_language, target_language):
    """
    Calcule les métriques d'entraînement du fine-tuning.
    
    Args:
        train_history: Historique d'entraînement du Trainer
        training_time: Temps d'entraînement en secondes
        source_language: Langue source
        target_language: Langue cible
        
    Returns:
        dict: Dictionnaire des métriques de training
    """
    
    print("🔄 Calcul des métriques de training...")
    
    # Extraire métriques d'entraînement
    train_losses = [log['train_loss'] for log in train_history if 'train_loss' in log]
    eval_losses = [log['eval_loss'] for log in train_history if 'eval_loss' in log]
    
    final_eval_loss = eval_losses[-1] if eval_losses else 0.0
    initial_eval_loss = eval_losses[0] if eval_losses else final_eval_loss
    best_eval_loss = min(eval_losses) if eval_losses else final_eval_loss
    
    # Trouver l'epoch du meilleur loss
    best_epoch = 1
    for i, log in enumerate(train_history):
        if 'eval_loss' in log and log['eval_loss'] == best_eval_loss:
            best_epoch = log.get('epoch', i + 1)
            break
    
    # Métriques de performance (approximation sans les vrais tokens)
    estimated_tokens = len(train_history) * 1000  # Estimation
    tokens_per_second = estimated_tokens / max(training_time, 1)
    
    # Mémoire GPU/MPS (placeholder)
    gpu_memory_peak = 0
    
    metrics = {
        "loss_metrics": {
            "final_train_loss": train_losses[-1] if train_losses else 0.0,
            "final_eval_loss": final_eval_loss,
            "best_eval_loss": best_eval_loss,
            "initial_eval_loss": initial_eval_loss,
            "loss_improvement": initial_eval_loss - best_eval_loss
        },
        "convergence_metrics": {
            "epochs_completed": len(set(log.get('epoch', 1) for log in train_history)),
            "early_stopped": best_epoch < len(set(log.get('epoch', 1) for log in train_history)),
            "convergence_epoch": best_epoch,
            "total_training_steps": len(train_history)
        },
        "performance_metrics": {
            "training_time_seconds": round(training_time, 2),
            "tokens_per_second": round(tokens_per_second, 0),
            "gpu_memory_peak_gb": round(gpu_memory_peak, 2),
            "total_tokens_processed": estimated_tokens
        }
    }
    
    print(f"📊 Métriques training: Loss amélioration={metrics['loss_metrics']['loss_improvement']:.4f}, "
          f"Temps={training_time:.1f}s, Epochs={metrics['convergence_metrics']['epochs_completed']}")
    
    return metrics

def calculate_translation_metrics(corpus_texts, batch_reference, source_language, target_language):
    """
    Calcule les métriques de traduction sur les textes déjà traduits.
    
    Args:
        corpus_texts: Dictionnaire contenant tous les versets avec leurs traductions
        batch_reference: Référence du batch à évaluer
        source_language: Langue source
        target_language: Langue cible
        
    Returns:
        dict: Dictionnaire des métriques de traduction
    """
    
    print("🔄 Calcul des métriques de traduction...")
    
    # Extraire les textes du batch
    all_source_texts = []
    all_reference_texts = []
    all_pretuned_texts = []
    all_posttuned_texts = []
    
    for verse_id, verse_data in corpus_texts.items():
        if (verse_id != "batch_data" and
            isinstance(verse_data, dict) and
            verse_data.get("batch_reference") == batch_reference):
            
            all_source_texts.append(verse_data.get('source_text', '').strip())
            all_reference_texts.append(verse_data.get('target_text', '').strip())
            all_pretuned_texts.append(verse_data.get('pretuned_target_text', '').strip())
            all_posttuned_texts.append(verse_data.get('posttuned_target_text', '').strip())
    
    if not all_source_texts:
        print("⚠️ Aucun verset trouvé pour ce batch")
        return {}
    
    # Limiter à 15 versets pour éviter les problèmes de RAM avec COMET. Les versets sont choisis de manière aléatoire dans le corpus
    max_verses = min(15, len(all_source_texts))
    
    if len(all_source_texts) > max_verses:
        # Échantillonnage aléatoire sans remise
        random.seed(42)
        random_indices = random.sample(range(len(all_source_texts)), max_verses)
        
        # Créer de nouvelles listes échantillonnées
        source_texts = [all_source_texts[i] for i in random_indices]
        reference_texts = [all_reference_texts[i] for i in random_indices]
        pretuned_texts = [all_pretuned_texts[i] for i in random_indices]
        posttuned_texts = [all_posttuned_texts[i] for i in random_indices]
        
        print(f"📊 Évaluation sur {max_verses} versets choisis aléatoirement (sur {len(all_source_texts)} total)")
    else:
        # Si on a 15 versets ou moins, on garde tout (comportement inchangé)
        source_texts = all_source_texts[:max_verses]
        reference_texts = all_reference_texts[:max_verses]
        pretuned_texts = all_pretuned_texts[:max_verses]
        posttuned_texts = all_posttuned_texts[:max_verses]
        
        print(f"📊 Évaluation sur {max_verses} versets (totalité du batch)")
    
    # Configuration selon la langue
    is_chinese = target_language == "Chinese"
    
    # Configuration ROUGE
    if is_chinese and ROUGE_CHINESE_AVAILABLE:
        print("📊 Utilisation de rouge-chinese pour évaluation chinoise")
        rouge_chinese_scorer = RougeChineseLib()
        use_rouge_chinese = True
    else:
        # Utiliser rouge-score standard (sans stemming pour le chinois)
        use_stemmer = not is_chinese
        rouge_standard_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=use_stemmer)
        use_rouge_chinese = False
        if is_chinese:
            print("📊 Utilisation de rouge-score standard pour évaluation chinoise (rouge_chinese pas disponible)")
    
    # Configuration BLEU
    smoothing = SmoothingFunction().method1
    
    # Calculer métriques pour pretuned et posttuned
    def calculate_metrics_for_translations(generated_texts, translation_type):
        bleu_scores = []
        rouge_scores = []
        meteor_scores = []
        comet_scores = []
        
        for i, (gen_text, ref_text, src_text) in enumerate(zip(generated_texts, reference_texts, source_texts)):
            if not gen_text or not ref_text:
                continue
                
            # Post-traitement chinois si nécessaire
            if is_chinese:
                gen_text = preserve_chinese_punctuation(gen_text)
                ref_text = preserve_chinese_punctuation(ref_text)
            
            # Nettoyer les textes
            gen_text = gen_text.strip()
            ref_text = ref_text.strip()
            src_text = src_text.strip()
            
            if gen_text and ref_text:
                # BLEU avec tokenisation appropriée
                if is_chinese:
                    gen_tokens = list(jieba.cut(gen_text))
                    ref_tokens = [list(jieba.cut(ref_text))]
                else:
                    gen_tokens = gen_text.split()
                    ref_tokens = [ref_text.split()]
                
                # Calcul BLEU
                if SACREBLEU_AVAILABLE and not is_chinese:
                    # SacreBLEU pour langues non-chinoises
                    bleu = sacrebleu.sentence_bleu(gen_text, [ref_text]).score / 100.0
                else:
                    # NLTK BLEU (avec smoothing pour phrases courtes)
                    bleu = sentence_bleu(ref_tokens, gen_tokens, smoothing_function=smoothing)
                
                bleu_scores.append(bleu)
                
                # ROUGE-L
                if use_rouge_chinese:
                    # Rouge-chinese nécessite une tokenisation préalable
                    gen_tokenized = ' '.join(jieba.cut(gen_text))
                    ref_tokenized = ' '.join(jieba.cut(ref_text))
                    rouge_result = rouge_chinese_scorer.get_scores(gen_tokenized, ref_tokenized)[0]
                    rouge_scores.append(rouge_result['rouge-l']['f'])
                else:
                    # Rouge-score standard
                    rouge = rouge_standard_scorer.score(ref_text, gen_text)
                    rouge_scores.append(rouge['rougeL'].fmeasure)
                
                # METEOR (seulement pour l'anglais)
                if target_language == "English":
                    try:
                        meteor = meteor_score([ref_text.split()], gen_text.split())
                        meteor_scores.append(meteor)
                    except:
                        meteor_scores.append(0.0)
                        
                # COMET
                if COMET_AVAILABLE and src_text:
                    comet_model = _get_comet_model()
                    if comet_model:
                        try:
                            # Préparer les données pour COMET
                            import os
                            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                            
                            comet_data = [{
                                "src": src_text,
                                "mt": gen_text,
                                "ref": ref_text
                            }]
                            comet_output = comet_model.predict(
                                comet_data,
                                batch_size=1,
                                accelerator="cpu"
                            )
                            
                            comet_scores.append(comet_output.scores[0])
                            
                        except Exception as e:
                            print(f"⚠️ Erreur COMET verset {i} ({translation_type}): {e}")
                            comet_scores.append(0.0)
                    else:
                        comet_scores.append(0.0)
            
        return {
            f"bleu_score": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
            f"rouge_l_score": sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0,
            f"meteor_score": sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0,
            f"comet_score": sum(comet_scores) / len(comet_scores) if comet_scores else 0.0,
            f"verses_evaluated": len(bleu_scores)
        }
    
    # Calculer métriques pour les deux types de traduction
    pretuned_metrics = calculate_metrics_for_translations(pretuned_texts, "pretuned")
    posttuned_metrics = calculate_metrics_for_translations(posttuned_texts, "posttuned")
    
    metrics = {
        "pretuned_quality_metrics": pretuned_metrics,
        "posttuned_quality_metrics": posttuned_metrics,
        "evaluation_method": {
            "rouge_lib": "rouge-chinese" if use_rouge_chinese else "rouge-score",
            "bleu_lib": "sacrebleu" if SACREBLEU_AVAILABLE and not is_chinese else "nltk",
            "tokenization": "character" if is_chinese else "word"
        }
    }
    
    eval_method = metrics['evaluation_method']
    pre_bleu = pretuned_metrics['bleu_score']
    post_bleu = posttuned_metrics['bleu_score']
    pre_rouge = pretuned_metrics['rouge_l_score']
    post_rouge = posttuned_metrics['rouge_l_score']
    pre_comet = pretuned_metrics['comet_score']
    post_comet = posttuned_metrics['comet_score']
    
    print(f"📊 Métriques traduction:")
    print(f"   Pretuned: BLEU={pre_bleu:.3f}, ROUGE-L={pre_rouge:.3f}", end="")
    if pre_comet is not None:
        print(f", COMET={pre_comet:.3f}")
    else:
        print()
    print(f"   Posttuned: BLEU={post_bleu:.3f}, ROUGE-L={post_rouge:.3f}", end="")
    if post_comet is not None:
        print(f", COMET={post_comet:.3f}")
    else:
        print()
    
    print(f"   Amélioration: BLEU={post_bleu-pre_bleu:+.3f}, ROUGE-L={post_rouge-pre_rouge:+.3f}", end="")
    if pre_comet is not None and post_comet is not None:
        print(f", COMET={post_comet-pre_comet:+.3f}")
    else:
        print()
        
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch.set_default_device('cpu')
    torch.set_default_tensor_type(torch.FloatTensor)
    
    return metrics

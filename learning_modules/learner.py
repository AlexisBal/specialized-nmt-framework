"""
Learner - Module d'apprentissage
Processus d'apprentissage :
1. Apprentissage sur les données réelles : source_text / target_text
2. Fine-tuning du modèle
3. Post-apprentissage par mécanisme d'attention
"""

import json
from datetime import datetime
from collections import Counter, defaultdict

# Imports des autres modules
from memory_manager.memorisator import save_batch_learning
from tuner.tuner_batch_summary import store_batch_metrics
from tuner.metrics_calculator import calculate_translation_metrics
from learning_modules.dictionary_learner import learn_from_dictionary_verse
from learning_modules.similarity_learner import learn_from_similarity_verse
from learning_modules.dominance_learner import learn_from_dominance_batch
from learning_modules.proper_noun_learner import learn_from_proper_noun_batch
from text_organizer.reorganize_texts import reorganize_verses_from_index
from memory_manager.cleaner import cleanup_intermediate_files
from learning_modules.learner_batch_summary import print_batch_learning_summary, get_batch_learning_totals
from tuner.tuner_batch_summary import print_batch_tuning_summary
from learning_modules.learner_all_summary import store_batch_learning_data, print_global_learning_summary
from tuner.translator import translate, translate_batch_texts
from tuner.finetuner import fine_tune_batch

def run_learning(book_range, semantic_dictionary, learned_dictionary, pinyin_dictionary, pivot_dictionary,
                source_language, target_language, base_model, learning_enabled, tuning_enabled):
    """Lance le pipeline d'apprentissage sur tous les batches du book_range.
    
    Returns:
        dict: global_stats avec total_learned, total_updated, tuning_metrics.
    """
    
    # Charger les données préparées
    filepath = f"outputs/translations/corpus_{book_range.replace('-', '_')}.json"
    with open(filepath, 'r', encoding='utf-8') as f:
        corpus_texts = json.load(f)
    
    # Identifier les batches uniques
    batch_references = set()
    for verse_data in corpus_texts.values():
        if isinstance(verse_data, dict) and 'batch_reference' in verse_data:
            batch_references.add(verse_data['batch_reference'])
    
    batch_refs_with_nums = []
    for verse_data in corpus_texts.values():
        if isinstance(verse_data, dict) and 'batch_reference' in verse_data:
            batch_ref = verse_data['batch_reference']
            book_num = int(batch_ref.split('_')[0])
            batch_num = verse_data['batch_number']
            if (book_num, batch_num, batch_ref) not in batch_refs_with_nums:
                batch_refs_with_nums.append((book_num, batch_num, batch_ref))
    batch_refs_with_nums.sort(key=lambda x: (x[0], x[1]))
    batch_references = [ref for book, batch, ref in batch_refs_with_nums]
    
    # Statistiques globales avec métriques de tuning
    global_stats = {
        'total_learned': 0,
        'total_updated': 0,
        'total_batches': len(batch_references),
        'batch_results': [],
        'tuning_metrics': {
            'avg_bleu': 0,
            'avg_rouge': 0,
            'total_loss_improvement': 0,
            'total_training_time': 0,
            'peak_memory': 0,
            'best_batch': None
        }
    }
    
    current_model = base_model
    tuning_scores = []
    
    # Apprendre par itération
    for i, batch_reference in enumerate(batch_references, 1):
        print(f"Traitement du batch {i}/{len(batch_references)}: {batch_reference}")
        
        batch_result = run_single_batch(
            corpus_texts=corpus_texts,
            batch_reference=batch_reference,
            book_range=book_range,
            semantic_dictionary=semantic_dictionary,
            pinyin_dictionary=pinyin_dictionary,
            learned_dictionary=learned_dictionary,
            pivot_dictionary=pivot_dictionary,
            source_language=source_language,
            target_language=target_language,
            base_model=current_model,
            learning_enabled=learning_enabled,
            tuning_enabled=tuning_enabled
        )
        
        # Accumuler les stats d'apprentissage
        global_stats['total_learned'] += batch_result['total_learned']
        global_stats['total_updated'] += batch_result['total_updated']
        global_stats['batch_results'].append({
            'batch_reference': batch_reference,
            'learned': batch_result['total_learned'],
            'updated': batch_result['total_updated']
        })
        
        # Accumuler les métriques de tuning si disponibles
        if 'training_metrics' in batch_result:
            metrics = batch_result['training_metrics']
            
            # Collecter les scores
            bleu = metrics.get('quality_metrics', {}).get('bleu_score', 0)
            rouge = metrics.get('quality_metrics', {}).get('rouge_l_score', 0)
            comet = metrics.get('quality_metrics', {}).get('comet_score', 0)
            loss_imp = metrics.get('loss_metrics', {}).get('loss_improvement', 0)
            training_time = metrics.get('performance_metrics', {}).get('training_time_seconds', 0)
            memory = metrics.get('performance_metrics', {}).get('gpu_memory_peak_gb', 0)
            
            tuning_scores.append({
                'bleu': bleu,
                'rouge': rouge,
                'comet': comet,
                'batch_ref': batch_reference
            })
            
            global_stats['tuning_metrics']['total_loss_improvement'] += loss_imp
            global_stats['tuning_metrics']['total_training_time'] += training_time
            global_stats['tuning_metrics']['peak_memory'] = max(
                global_stats['tuning_metrics']['peak_memory'], memory
            )
        
        # Mettre à jour le modèle si le fine-tuning est réussi
        if tuning_enabled and 'finetuned_model_path' in batch_result and batch_result['finetuned_model_path']:
            current_model = batch_result['finetuned_model_path']
    
    # Calculer les moyennes des métriques de tuning
    if tuning_scores:
        global_stats['tuning_metrics']['avg_bleu'] = sum(s['bleu'] for s in tuning_scores) / len(tuning_scores)
        global_stats['tuning_metrics']['avg_rouge'] = sum(s['rouge'] for s in tuning_scores) / len(tuning_scores)
        global_stats['tuning_metrics']['avg_comet'] = sum(s['comet'] for s in tuning_scores) / len(tuning_scores)
        
        # Identifier le meilleur batch
        best = max(tuning_scores, key=lambda x: x['comet'] if x['comet'] > 0 else x['rouge'])
        global_stats['tuning_metrics']['best_batch'] = {
            'name': best['batch_ref'],
            'bleu': best['bleu'],
            'rouge': best['rouge'],
            'comet': best['comet']
        }
    
    if tuning_enabled:
        global_stats['final_model'] = current_model
    else:
        global_stats['final_model'] = None
        
    print(f"Apprentissage terminé sur {len(batch_references)} batches")

    # Rapport final de tuning
    if tuning_enabled:
        try:
            from tuner.tuner_all_summary import print_global_tuning_summary
            print_global_tuning_summary()
        except Exception as e:
            print(f"⚠️ Erreur rapport final tuning: {e}")
            
    # Rapport final d'apprentissage
    if learning_enabled:
        try:
            print_global_learning_summary()
        except Exception as e:
            print(f"⚠️ Erreur rapport final apprentissage: {e}")
    
    return global_stats

def run_single_batch(
    corpus_texts,               # JSON avec tous les versets
    batch_reference,            # ex: "1_1_1-4_13" pour identifier les versets du batch
    book_range,                 # ex: "1_3" pour identifier le book_range (sauvegarde du fichier json)
    semantic_dictionary,        # {source_term: [target_term_candidates]}
    pinyin_dictionary,          # {source_term: {target_term: pinyin}} pour chinois
    learned_dictionary,         # dictionnaire des termes déjà appris
    pivot_dictionary,           # dictionnaire pivot si langues anciennes
    source_language,            # "English", "French", etc.
    target_language,            # "Chinese", "Spanish", etc.
    base_model,                 # modèle de base à utiliser
    learning_enabled,           # Activer apprentissage
    tuning_enabled              # Activer tuning
):
    """
    Pipeline principal d'apprentissage pour un batch complet
    """
    
    print(f"🚀 Début apprentissage batch: {batch_reference}")    
    # Variables de tracking pour toutes les étapes
    total_dict_learned = []
    total_dict_updated = []
    total_similarity_learned = []
    total_similarity_updated = []
    
    # Accumulation des cooccurrences pour les étapes de batch
    cooccurrences_batch = defaultdict(Counter)
    
    # Construction des textes complets du batch
    batch_source_text = ""
    batch_target_text = ""
    
    # APPRENTISSAGE SUR TEXTES SOURCE ET TARGET
    
    # Initialiser verse_count pour tous les cas
    verse_count = 0
    for verse_reference, verse_data in corpus_texts.items():
        if isinstance(verse_data, dict) and verse_data.get("batch_reference") == batch_reference:
            verse_count += 1
            # Construction des textes batch (toujours nécessaire)
            batch_source_text += verse_data['source_text'] + " "
            batch_target_text += verse_data['target_text'] + " "
    
    if learning_enabled:
        # 1. ÉTAPE VERSET PAR VERSET
        for verse_reference, verse_data in corpus_texts.items():
            # Vérifier si le verset appartient au batch
            if isinstance(verse_data, dict) and verse_data.get("batch_reference") == batch_reference:
                
                # 1.1 Apprentissage par dictionnaire
                dict_result = learn_from_dictionary_verse(
                    semantic_dictionary=semantic_dictionary,
                    verse_reference=verse_reference,
                    corpus_texts=corpus_texts,
                    learned_dictionary=learned_dictionary,
                    source_language=source_language,
                    target_language=target_language,
                    pivot_dictionary=pivot_dictionary
                )
                
                # Collecter les résultats du dictionnaire
                total_dict_learned.extend(dict_result['dict_verse_learned_terms'])
                total_dict_updated.extend(dict_result['dict_verse_updated_terms'])
                
                # 1.2 Apprentissage par similarité
                similarity_result = learn_from_similarity_verse(
                    verse_reference=verse_reference,
                    cleaned_cooccurrences=dict_result['cleaned_cooccurrences'],
                    semantic_dictionary=semantic_dictionary,
                    pinyin_dictionary=pinyin_dictionary,
                    learned_dictionary=learned_dictionary,
                    source_language=source_language,
                    target_language=target_language
                )
                
                # Collecter les résultats de similarité
                total_similarity_learned.extend(similarity_result['similarity_verse_learned_terms'])
                total_similarity_updated.extend(similarity_result['similarity_verse_updated_terms'])
                
                # 1.3 Accumulation des cooccurrences restantes
                for source_term, target_counts in similarity_result['cleaned_cooccurrences'].items():
                    for target_term, count in target_counts.items():
                        cooccurrences_batch[source_term][target_term] += count
                
                # 1.4 Les textes batch sont déjà construits plus haut
                pass
        
        # Ajouter les textes batch au corpus
        corpus_texts["batch_data"] = {
            "batch_reference": batch_reference,
            "batch_source_text": batch_source_text.strip(),
            "batch_target_text": batch_target_text.strip(),
            "verse_count": verse_count,
            "created_timestamp": datetime.now().isoformat()
        }
        
        # 2. APPRENTISSAGE PAR DOMINANCE SUR LE BATCH
        dominance_result = learn_from_dominance_batch(
            cooccurrences_batch=cooccurrences_batch,
            batch_reference=batch_reference,
            learned_dictionary=learned_dictionary,
            source_language=source_language,
            target_language=target_language
        )
        
        # 3. APPRENTISSAGE DES NOMS PROPRES DU BATCH
        proper_nouns_result = learn_from_proper_noun_batch(
            batch_source_text=batch_source_text,
            batch_target_text=batch_target_text,
            cooccurrences_batch=cooccurrences_batch,
            cooccurrences_proper_batch=dominance_result['cleaned_proper_cooccurrences'],
            batch_reference=batch_reference,
            learned_dictionary=learned_dictionary,
            source_language=source_language,
            target_language=target_language
        )
        
        total_learned = (len(total_dict_learned) + len(total_similarity_learned) +
                        len(dominance_result['dominance_batch_learned_terms']) +
                        len(proper_nouns_result['proper_noun_learned_terms']))
        
        total_updated = (len(total_dict_updated) + len(total_similarity_updated) +
                        len(dominance_result['dominance_batch_updated_terms']) +
                        len(proper_nouns_result['proper_noun_updated_terms']))
                        
    else:
        # Pas d'apprentissage, initialiser les variables
        total_learned = 0
        total_updated = 0
        dominance_result = {'dominance_batch_learned_terms': [], 'dominance_batch_updated_terms': []}
        proper_nouns_result = {'proper_noun_learned_terms': [], 'proper_noun_updated_terms': [], 'proper_noun_unresolved': []}
        
        # Ajouter les textes batch au corpus même sans apprentissage
        corpus_texts["batch_data"] = {
            "batch_reference": batch_reference,
            "batch_source_text": batch_source_text.strip(),
            "batch_target_text": batch_target_text.strip(),
            "verse_count": verse_count,
            "created_timestamp": datetime.now().isoformat()
        }
                        
    # TRADUCTION INITIALE, TUNING DU MODÈLE ET TRADUCTION FINALE
    if tuning_enabled:
    
        # BLOC DE TRADUCTION AVEC LE MODÈLE PRÉ-TUNÉ DE BASE
        # 1. Rassembler les textes
        texts_to_translate = []
        verse_ids_ordered = []

        for verse_id, verse_data in corpus_texts.items():
            if (verse_id != "batch_data" and isinstance(verse_data, dict) and verse_data.get("batch_reference") == batch_reference):
                texts_to_translate.append(verse_data['source_text'])
                verse_ids_ordered.append(verse_id)

        # 2. Traduire avec le modèle pré-tuné
        if texts_to_translate:
            print(f"📝 Traduction de {len(texts_to_translate)} versets en batch...")
            translations = translate_batch_texts(
                model_path=base_model,
                source_texts=texts_to_translate,
                source_language=source_language,
                target_language=target_language
            )
            
            # S'assurer que translations n'est pas None
            if translations is None:
                print("❌ translate_batch_texts a retourné None")
                translations = [""] * len(texts_to_translate)  # Fallback
            elif len(translations) != len(texts_to_translate):
                print(f"⚠️ Nombre de traductions ({len(translations)}) ≠ textes ({len(texts_to_translate)})")
                # Compléter avec des chaînes vides si nécessaire
                while len(translations) < len(texts_to_translate):
                    translations.append("")
            
            # 3. Répartir les traductions pré-tunées
            for i, verse_id in enumerate(verse_ids_ordered):
                if i < len(translations):
                    corpus_texts[verse_id]["pretuned_target_text"] = translations[i] or ""
                else:
                    corpus_texts[verse_id]["pretuned_target_text"] = ""
            
            print(f"✅ Traduction batch terminée: {len(translations)} traductions")
                        
        else:
            print("⚠️ Aucun texte à traduire trouvé")

        # FINE-TUNING SUR LE BATCH
        
        batch_number = batch_reference.split('_')[0] if '_' in batch_reference else batch_reference
        
        result = fine_tune_batch(
            model_path=base_model,
            source_text=batch_source_text,
            target_text=batch_target_text,
            batch_number=batch_number,
            source_language=source_language,
            target_language=target_language,
            corpus_texts=corpus_texts,
            batch_reference=batch_reference
        )
        
        if isinstance(result, tuple):
            finetuned_model_path, training_metrics = result
        else:
            finetuned_model_path = result
            training_metrics = {}
            
        # Récupérer le batch_number depuis les données des versets
        batch_number = None
        for verse_data in corpus_texts.values():
            if isinstance(verse_data, dict) and verse_data.get("batch_reference") == batch_reference:
                batch_number = verse_data.get("batch_number")
                break
        # Fallback si pas trouvé
        if batch_number is None:
            batch_number = batch_reference
        
        # AFFICHER LES MÉTRIQUES DE TUNING DU BATCH
        if finetuned_model_path and training_metrics:
            try:
                print_batch_tuning_summary(batch_number, training_metrics, source_language, target_language)
                store_batch_metrics(batch_number, training_metrics, source_language, target_language)
            except Exception as e:
                print(f"⚠️ Erreur affichage métriques tuning: {e}")

            # GÉNÉRER DES TRADUCTIONS AVEC LE MODÈLE TUNÉ
            post_tuning_texts = []
            post_tuning_verse_ids = []

            for verse_id, verse_data in corpus_texts.items():
                if (verse_id != "batch_data" and isinstance(verse_data, dict) and verse_data.get("batch_reference") == batch_reference):
                    post_tuning_texts.append(verse_data['source_text'])
                    post_tuning_verse_ids.append(verse_id)

            if post_tuning_texts:
                try:
                    print(f"📝 Traduction post-tuning de {len(post_tuning_texts)} versets...")
                    post_translations = translate_batch_texts(
                        model_path=finetuned_model_path,
                        source_texts=post_tuning_texts,
                        source_language=source_language,
                        target_language=target_language
                    )
                    
                    # Répartir les traductions post-tuning
                    for i, verse_id in enumerate(post_tuning_verse_ids):
                        if i < len(post_translations):
                            corpus_texts[verse_id]["posttuned_target_text"] = post_translations[i] or ""
                        else:
                            corpus_texts[verse_id]["posttuned_target_text"] = ""
                    
                    print(f"✅ Traduction post-tuning terminée: {len(post_translations)} traductions")
                
                except Exception as e:
                    print(f"❌ Erreur traduction post-tuning batch: {e}")
                    # Fallback individuel si nécessaire
                    for verse_id in post_tuning_verse_ids:
                        corpus_texts[verse_id]["posttuned_target_text"] = ""
        else:
            print("❌ Fine-tuning échoué, pas de traduction post-tuning")
            finetuned_model_path = base_model  # Fallback sur le modèle de base
            
    else:
        # Pas de tuning, pas de traduction
        finetuned_model_path = base_model
        training_metrics = {}
        print("⏭️ Tuning désactivé, pas de traduction")
        
    # SAUVEGARDER LE CORPUS DES TEXTES
    try:
        corpus_filepath = f"outputs/translations/corpus_{book_range.replace('-', '_')}.json"
        with open(corpus_filepath, 'w', encoding='utf-8') as f:
            json.dump(corpus_texts, f, ensure_ascii=False, indent=2)
        print(f"💾 Corpus sauvegardé: {corpus_filepath}")
    except Exception as e:
        print(f"❌ Erreur sauvegarde corpus: {e}")
    
    # NETTOYER LES FICHIERS INTERMÉDIAIRES
    if tuning_enabled:
        try:
            cleanup_intermediate_files(model_path=finetuned_model_path)
        except Exception as e:
            print(f"⚠️ Erreur nettoyage fichiers: {e}")
    
    # AFFICHER LES DONNÉES D'APPRENTISSAGE
    if learning_enabled:
        try:
            print_batch_learning_summary(
                batch_reference=batch_reference,
                verse_count=verse_count,
                total_dict_learned=total_dict_learned,
                total_dict_updated=total_dict_updated,
                total_similarity_learned=total_similarity_learned,
                total_similarity_updated=total_similarity_updated,
                dominance_result=dominance_result,
                proper_nouns_result=proper_nouns_result
            )
        except Exception as e:
            print(f"⚠️ Erreur affichage résumé batch: {e}")
        
    # SAUVEGARDER LE DICTIONNAIRE APPRIS
    if learning_enabled:
        try:
            save_batch_learning(
                learned_dictionary=learned_dictionary,
                cumulative_cooccurrences={},  # Sera géré par la fonction
                source_text=batch_source_text,
                target_text=batch_target_text,
                batch_number=batch_reference,
                verses_processed=verse_count,
                learned_dictionary_path="outputs/dictionaries/learned_dictionary.json"
            )
        except Exception as e:
            print(f"❌ Erreur sauvegarde dictionnaire appris: {e}")
    else:
        print("⏭️ Sauvegarde du dictionnaire sautée (learning désactivé)")
      
    # SAUVEGARDER LES DONNÉES D'APPRENTISSAGE
    if learning_enabled:
        try:
            store_batch_learning_data(
                batch_reference=batch_reference,
                verse_count=verse_count,
                total_dict_learned=total_dict_learned,
                total_similarity_learned=total_similarity_learned,
                dominance_result=dominance_result,
                proper_nouns_result=proper_nouns_result
            )
        except Exception as e:
            print(f"⚠️ Erreur stockage données apprentissage: {e}")
    else:
        print("⏭️ Sauvegarde des données d'apprentissage sautée (learning désactivé)")
        
    # NETTOYER LES TEXTES POUR LE BATCH SUIVANT
    corpus_texts["batch_data"] = {
        "batch_reference": batch_reference,
        "verse_count": verse_count,
        "created_timestamp": datetime.now().isoformat()
        # Les textes batch sont supprimés
    }
    
    # CALCULER LES MÉTRIQUES DE TRADUCTION
    if tuning_enabled:
    
        # LIBÉRER la RAM pour COMET
        print("🧹 Libération maximale RAM CPU pour métriques COMET...")
        
        # Supprimer toutes les variables lourdes qui ne sont pas retournées
        import gc
        import torch
        import os
        
        deleted_count = 0
        
        # MÉTHODE CORRECTE: Supprimer chaque variable individuellement
        if 'batch_source_text' in locals():
            del batch_source_text
            deleted_count += 1
        if 'batch_target_text' in locals():
            del batch_target_text
            deleted_count += 1
        if 'cooccurrences_batch' in locals():
            del cooccurrences_batch
            deleted_count += 1
        if 'texts_to_translate' in locals():
            del texts_to_translate
            deleted_count += 1
        if 'verse_ids_ordered' in locals():
            del verse_ids_ordered
            deleted_count += 1
        if 'translations' in locals():
            del translations
            deleted_count += 1
        if 'post_tuning_texts' in locals():
            del post_tuning_texts
            deleted_count += 1
        if 'post_tuning_verse_ids' in locals():
            del post_tuning_verse_ids
            deleted_count += 1
        if 'post_translations' in locals():
            del post_translations
            deleted_count += 1
        if 'result' in locals():
            del result
            deleted_count += 1
        if 'batch_number' in locals():
            del batch_number
            deleted_count += 1
            
        print(f"🗑️ Supprimé {deleted_count} variables")
        
        # Forcer garbage collection (efficace sur CPU)
        for i in range(3):
            collected = gc.collect()
            if collected > 0:
                print(f"🔄 GC pass {i+1}: {collected} objets libérés")
        
        # Nettoyer caches PyTorch CPU
        torch.clear_autocast_cache()
        if hasattr(torch.backends, 'mkldnn'):
            torch.backends.mkldnn.enabled = False
            torch.backends.mkldnn.enabled = True
        
        # Nettoyer caches tokenizers
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Monitoring mémoire (optionnel)
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"💾 RAM utilisée après nettoyage: {memory_mb:.1f} MB")
        except:
            pass
        
        print("✅ Nettoyage RAM maximal terminé")
        
        # CALCULER les métriques de traduction
        try:
            print("📊 Calcul des métriques de traduction...")
            translation_metrics = calculate_translation_metrics(
                corpus_texts=corpus_texts,
                batch_reference=batch_reference,
                source_language=source_language,
                target_language=target_language
            )
            
            # Combiner avec les métriques de training
            if training_metrics:
                training_metrics.update(translation_metrics)
                if i == 1:  # Premier batch
                    from tuner.tuner_batch_summary import store_batch_1_baseline
                    store_batch_1_baseline(training_metrics)
            else:
                training_metrics = translation_metrics
                
        except Exception as e:
            print(f"❌ Erreur calcul métriques traduction: {e}")
            
    return {
        'batch_reference': batch_reference,
        'verse_count': verse_count,
        'dictionary_results': {
            'learned': total_dict_learned,
            'updated': total_dict_updated
        },
        'similarity_results': {
            'learned': total_similarity_learned,
            'updated': total_similarity_updated
        },
        'dominance_results': {
            'learned': dominance_result['dominance_batch_learned_terms'],
            'updated': dominance_result['dominance_batch_updated_terms']
        },
        'proper_noun_results': {
            'learned': proper_nouns_result['proper_noun_learned_terms'],
            'updated': proper_nouns_result['proper_noun_updated_terms'],
            'unresolved': proper_nouns_result['proper_noun_unresolved']
        },
        'total_learned': total_learned,
        'total_updated': total_updated,
        'finetuned_model_path': finetuned_model_path,
        'training_metrics': training_metrics
    }

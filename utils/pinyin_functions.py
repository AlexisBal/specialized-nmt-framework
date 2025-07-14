"""
Utils pour le calcul de la similarité pinyin pour le chinois
"""

import difflib, re
from pypinyin import lazy_pinyin, Style

def get_pinyin(chinese_word, pinyin_dict=None):
    """
    Récupère le pinyin d'un mot chinois
    
    Args:
        chinese_word: Mot chinois à analyser
        pinyin_dict: Dictionnaire optionnel {english_term: {chinese_term: pinyin}}
    
    Returns:
        str: Pinyin du mot (ex: "sai1 te4")
    """
    # 1. Essayer le dictionnaire de référence s'il est fourni
    if pinyin_dict:
        for english_term, chinese_pinyin_map in pinyin_dict.items():
            if chinese_word in chinese_pinyin_map:
                return chinese_pinyin_map[chinese_word]
    
    # 2. Utiliser lazy_pinyin directement
    result = lazy_pinyin(chinese_word, style=Style.TONE3)
    return ' '.join(result).capitalize()  # "sai1 te4"

def calculate_similarity(word1, word2, pinyin1, pinyin2):
    """
    Calcule la similarité entre deux mots chinois
    
    Args:
        word1, word2: Mots chinois à comparer
        pinyin1, pinyin2: Pinyin des mots
    
    Returns:
        float: Score de similarité entre 0 et 1
    """
    # 1. Similarité de caractères (40%)
    char_sim = 0.0
    if len(word1) == len(word2) and len(word1) > 0:
        matches = sum(1 for c1, c2 in zip(word1, word2) if c1 == c2)
        char_sim = matches / len(word1)
    
    # 2. Similarité pinyin (40%)
    pinyin_sim = 0.0
    if pinyin1 and pinyin2:
        pinyin_sim = difflib.SequenceMatcher(None, pinyin1.lower(), pinyin2.lower()).ratio()
    
    # 3. Pénalité différence de longueur (20%)
    len_penalty = 1.0
    if len(word1) > 0 and len(word2) > 0:
        len_diff = abs(len(word1) - len(word2))
        max_len = max(len(word1), len(word2))
        len_penalty = 1.0 - (len_diff / max_len) * 0.5  # Pénalité progressive
        len_penalty = max(0.0, len_penalty)
    
    # 4. Score combiné pondéré
    final_score = (char_sim * 0.4 + pinyin_sim * 0.4 + len_penalty * 0.2)
    
    return final_score


def calculate_name_similarity(source_name, pinyin, source_language="English"):
    """
    Calcule la similarité phonétique pour noms propres multilingues-chinois
    
    Args:
        source_name: Nom dans la langue source
        pinyin: Pinyin du nom chinois
        source_language: Langue source ("English", "Hebrew", "Greek", "Latin", etc.)
    """
    
    # Nettoyer le pinyin (enlever les tons et espaces)
    clean_pinyin = re.sub(r'[0-9]', '', pinyin.lower()).replace(' ', '')
    source_clean = source_name.lower()
    
    # Longueur minimale de correspondance
    if len(source_clean) < 2 or len(clean_pinyin) < 2:
        return 0.0
    
    # 1. MAPPINGS PHONÉTIQUES PAR LANGUE
    def get_phonetic_mapping(language):
        """Retourne le mapping phonétique selon la langue source"""
        
        if language == "English":
            return {
                # Consonnes de base
                'br': 'bo', 'ham': 'han', 'a': 'ya', 'e': 'yi',
                
                # Phonétique générale
                'j': 'y', 'c': 'k', 'ch': 'ke', 'ck': 'ke', 'x': 'kesi',
                'th': 't', 'ph': 'f',
                
                # Groupes consonantiques
                'br': 'bo', 'gr': 'ge', 'pr': 'pu', 'tr': 'te',
                'dr': 'de', 'fr': 'fu', 'cr': 'ke',
                
                # Voyelles adaptées
                'aa': 'a', 'ee': 'i', 'oo': 'u', 'ou': 'ao',
                'ow': 'ao', 'ey': 'ei', 'ay': 'ai', 'oy': 'ao',
                
                # Sons difficiles
                'v': 'w',
                
                # Simplifications
                'bb': 'b', 'cc': 'k', 'dd': 'd', 'ff': 'f', 'gg': 'g',
                'll': 'l', 'mm': 'm', 'nn': 'n', 'pp': 'p', 'rr': 'r',
                'ss': 's', 'tt': 't'
            }
        
        elif language == "Hebrew":
            return {
                # Consonnes hébraïques courantes
                'sh': 's', 'ch': 'he', 'kh': 'he', 'ts': 'ci',
                'tz': 'ci', 'ph': 'f',
                
                # Voyelles hébraïques
                'ah': 'a', 'eh': 'ai', 'ih': 'i', 'oh': 'ao', 'uh': 'u',
                'ai': 'ai', 'ei': 'ai', 'oi': 'ao',
                
                # Terminaisons hébraïques courantes
                'el': 'ai', 'al': 'a', 'ah': 'a', 'im': 'mu',
                'ot': 'te', 'et': 'te', 'it': 'te',
                
                # Sons spécifiques
                'h': '', 'q': 'ke', 'w': 'wo', 'y': 'ya',
                
                # Simplifications
                'bb': 'b', 'dd': 'd', 'gg': 'g', 'll': 'l',
                'mm': 'm', 'nn': 'n', 'pp': 'p', 'rr': 'r', 'tt': 't'
            }
        
        elif language == "Greek":
            return {
                # Consonnes grecques
                'ch': 'ke', 'ph': 'f', 'th': 't', 'ps': 's', 'ks': 'ke',
                'gh': 'ge', 'kh': 'ke', 'rh': 'r',
                
                # Voyelles grecques
                'ai': 'ai', 'ei': 'ai', 'oi': 'ao', 'au': 'ao', 'eu': 'ao',
                'ou': 'u', 'ui': 'wei',
                
                # Terminaisons grecques
                'os': 'si', 'es': 'si', 'is': 'si', 'as': 'si',
                'ion': 'ang', 'tion': 'xiong',
                
                # Simplifications
                'bb': 'b', 'dd': 'd', 'gg': 'g', 'll': 'l',
                'mm': 'm', 'nn': 'n', 'pp': 'p', 'rr': 'r', 'tt': 't'
            }
        
        elif language == "Latin":
            return {
                # Consonnes latines
                'qu': 'kuo', 'c': 'k', 'ph': 'f', 'th': 't', 'ch': 'ke',
                'x': 'ke', 'j': 'ya',
                
                # Voyelles latines
                'ae': 'ai', 'oe': 'ao', 'au': 'ao', 'eu': 'ao',
                
                # Terminaisons latines
                'us': 'si', 'um': 'mu', 'is': 'si', 'es': 'si',
                'ius': 'si', 'tion': 'xiong', 'tio': 'xiao',
                
                # Simplifications
                'bb': 'b', 'cc': 'k', 'dd': 'd', 'ff': 'f', 'gg': 'g',
                'll': 'l', 'mm': 'm', 'nn': 'n', 'pp': 'p', 'rr': 'r', 'tt': 't'
            }
        
        else:
            # Mapping générique pour les autres langues
            return {
                'ch': 'ke', 'sh': 's', 'th': 't', 'ph': 'f',
                'qu': 'kuo', 'x': 'ke', 'j': 'ya', 'w': 'wo',
                'bb': 'b', 'cc': 'k', 'dd': 'd', 'ff': 'f', 'gg': 'g',
                'll': 'l', 'mm': 'm', 'nn': 'n', 'pp': 'p', 'rr': 'r', 'tt': 't'
            }
    
    phonetic_map = get_phonetic_mapping(source_language)
    
    # 2. RÈGLES POSITIONNELLES (adaptées par langue)
    def apply_smart_mapping(text, language):
        """Applique des règles intelligentes selon la position et la langue"""
        result = text
        
        if language == "English":
            # Préfixes silencieux anglais
            if result.startswith('kn'): result = result[2:]
            if result.startswith('ps'): result = result[1:]
            if result.startswith('wr'): result = result[1:]
            
            # Suffixes anglais
            if result.endswith('mb'): result = result[:-1]
            if result.endswith('gn'): result = result[:-2] + 'n'
            if result.endswith('bt'): result = result[:-2] + 't'
        
        elif language == "Hebrew":
            # Simplifications hébraïques
            if result.endswith('ah'): result = result[:-1]
            if result.endswith('el'): result = result[:-2] + 'l'
            if result.startswith('h'): result = result[1:]  # H initial souvent silencieux
        
        elif language == "Greek":
            # Simplifications grecques
            if result.endswith('os'): result = result[:-2] + 's'
            if result.endswith('es'): result = result[:-2] + 's'
            if result.startswith('ps'): result = result[1:]
        
        return result
    
    # 3. GÉNÉRER DES VARIANTES DE TRANSCRIPTION
    mapped_source = apply_smart_mapping(source_clean, source_language)
    
    def generate_transcription_variants(text, phonetic_map, max_variants=5):
        """Génère plusieurs variantes de transcription possibles"""
        variants = [text]
        
        for source_sound, target_sound in sorted(phonetic_map.items(), key=len, reverse=True):
            if source_sound in text:
                new_variants = []
                for variant in variants[:max_variants]:
                    new_variants.append(variant)
                    new_variant = variant.replace(source_sound, target_sound)
                    if new_variant != variant:
                        new_variants.append(new_variant)
                variants = new_variants[:max_variants]
        
        return variants
    
    source_variants = generate_transcription_variants(mapped_source, phonetic_map)
    
    # 4. CALCUL DE SIMILARITÉ SIMPLIFIÉ POUR NON-ANGLAIS
    if source_language != "English":
        # Pour les langues anciennes, calcul plus simple et direct
        best_similarity = 0.0
        
        for variant in source_variants:
            # Similarité directe
            direct_sim = difflib.SequenceMatcher(None, variant, clean_pinyin).ratio()
            
            # Bonus longueur appropriée
            length_ratio = min(len(variant), len(clean_pinyin)) / max(len(variant), len(clean_pinyin))
            length_bonus = 0.1 if length_ratio > 0.7 else 0.0
            
            # Score combiné simple
            combined_score = direct_sim + length_bonus
            best_similarity = max(best_similarity, combined_score)
        
        return min(1.0, best_similarity)
    
    # 5. CALCUL COMPLET POUR L'ANGLAIS
    def get_syllables(text, min_len=2):
        """Extraire des syllabes approximatives"""
        syllables = []
        for i in range(len(text) - min_len + 1):
            for length in [3, 2]:
                if i + length <= len(text):
                    syl = text[i:i+length]
                    syllables.append(syl)
        return set(syllables)
    
    best_scores = {'direct': 0.0, 'syllable': 0.0, 'length_ratio': 0.0}
    
    for variant in source_variants:
        # A. Similarité directe
        direct_sim = difflib.SequenceMatcher(None, variant, clean_pinyin).ratio()
        
        # B. Correspondances syllabiques
        variant_syllables = get_syllables(variant)
        pinyin_syllables = get_syllables(clean_pinyin)
        
        syllable_intersection = len(variant_syllables & pinyin_syllables)
        syllable_union = len(variant_syllables | pinyin_syllables)
        syllable_sim = syllable_intersection / syllable_union if syllable_union > 0 else 0
        
        # C. Ratio de longueur
        length_ratio = min(len(variant), len(clean_pinyin)) / max(len(variant), len(clean_pinyin))
        
        # Garder les meilleurs scores
        best_scores['direct'] = max(best_scores['direct'], direct_sim)
        best_scores['syllable'] = max(best_scores['syllable'], syllable_sim)
        best_scores['length_ratio'] = max(best_scores['length_ratio'], length_ratio)
    
    # Calculs finaux pour l'anglais
    direct_similarity = best_scores['direct']
    syllable_similarity = best_scores['syllable']
    length_ratio = best_scores['length_ratio']
    
    # Pénalité longueur
    length_penalty = 1.0
    min_length_diff = min(abs(len(variant) - len(clean_pinyin)) for variant in source_variants)
    
    if min_length_diff > 3:
        length_penalty = 0.5
    elif min_length_diff > 5:
        length_penalty = 0.2
    
    # Seuil cohérence phonétique
    min_phonetic_threshold = max(direct_similarity, syllable_similarity * 1.2)
    if min_phonetic_threshold < 0.25:
        return 0.0
    
    # Score final anglais
    final_score = (
        direct_similarity * 0.5 +
        syllable_similarity * 0.3 +
        length_ratio * 0.1 +
        min_phonetic_threshold * 0.1
    ) * length_penalty
    
    return min(1.0, final_score)

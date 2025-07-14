"""
Extractor
Extrait les caractères chinois pour un texte donné
Version avec Stanza + fallback vers méthode brute si échec
"""
import re

# Cache global pour éviter de recharger le modèle Stanza à chaque appel
_stanza_nlp = None
_stanza_available = None
_extracted_proper_nouns = set()

def extract_chinese_words(text):
    """
    Extraction des mots chinois avec Stanza, fallback vers méthode brute
    Garde la même signature que la version originale
    """
    global _stanza_nlp, _stanza_available
    
    # Tentative Stanza en priorité
    if _stanza_available is None:  # Premier appel
        try:
            import stanza
            # Initialiser le modèle Stanza une seule fois
            _stanza_nlp = stanza.Pipeline('zh-hant', processors='tokenize', verbose=False)
            _stanza_available = True
            print("✅ Stanza initialisé pour extraction chinoise")
        except Exception as e:
            print(f"⚠️ Stanza non disponible: {e}")
            print("🔄 Fallback vers méthode brute")
            _stanza_available = False
    
    # Utiliser Stanza si disponible
    if _stanza_available and _stanza_nlp is not None:
        try:
            return _extract_with_stanza(text)
        except Exception as e:
            print(f"⚠️ Erreur Stanza: {e}, fallback vers méthode brute")
            return _extract_chinese_words_brute(text)
    
    # Fallback vers méthode brute
    return _extract_chinese_words_brute(text)


def _extract_with_stanza(text):
    """Extraction avec Stanza (méthode optimale)"""
    global _stanza_nlp
    
    doc = _stanza_nlp(text)
    words = set()
    proper_nouns = set()  # Identifier les noms propres
    
    for sentence in doc.sentences:
        for token in sentence.tokens:
            word = token.text.strip()
            
            # Appliquer les mêmes filtres que la méthode brute pour cohérence
            if (len(word) >= 1 and
                not word.isdigit() and
                not re.search(r'[，。：；！？「」（）、""''…—·]', word) and
                not word.startswith(('是', '就', '的', '了', '又')) and
                not word.endswith(('是', '就', '的', '了', '又')) and
                not any(c in '的了是在就也都' for c in word)):
                words.add(word)
                
            # Vérifier si c'est un nom propre
            for sent_word in sentence.words:
                if sent_word.text == word and sent_word.upos == 'PROPN':
                    proper_nouns.add(word)
                    break
    
    # Stocker les noms propres dans une variable globale
    global _extracted_proper_nouns
    _extracted_proper_nouns = proper_nouns
    
    return words


def _extract_chinese_words_brute(text):
    """
    Méthode brute originale (fallback)
    Extraction simple des mots chinois par n-grammes
    """
    chars = list(text)
    words = set()
    _extracted_proper_nouns = set()
    
    # Extraire tous les segments de 1 à 4 caractères
    for i in range(len(chars)):
        for length in [2, 3, 4, 1]:
            if i + length <= len(chars):
                word = ''.join(chars[i:i+length]).strip()
                
                # Filtres simples et universels
                if (len(word) >= 1 and  # Garder même les caractères uniques
                    not word.isdigit() and  # Pas de nombres
                    ' ' not in word and  # Pas d'espaces
                    not re.search(r'[，。：；！？「」（）、""''…—·]', word) and  # Ponctuation complète
                    not word.startswith(('是', '就', '的', '了', '又')) and
                    not word.endswith(('是', '就', '的', '了', '又')) and
                    not any(c in '的了是在就也都' for c in word)):  # filtre les caractères grammaticaux les plus fréquents
                    words.add(word)
    return words

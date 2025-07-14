"""
Lemmatiseur
Lemmatise un texte donné en utilisant Stanza avec fallback vers spaCy, CLTK pour le grec (d'expérience meilleur que Stanza pour la gestion du grec biblique)
"""
import spacy

# Cache global pour éviter de recharger les modèles à chaque appel
_stanza_pipelines = {}
_spacy_pipelines = {}
_stanza_available = None
try:
    from cltk.lemmatize.grc import GreekBackoffLemmatizer
    from cltk.alphabet.text_normalization import cltk_normalize
    _cltk_available = True
    _cltk_greek_lemmatizer = None
except ImportError:
    _cltk_available = False
    _cltk_greek_lemmatizer = None
try:
    from cltk.tag import ner
    _cltk_ner_available = True
except ImportError:
    _cltk_ner_available = False
_cltk_greek_proper_nouns = {}

def extract_lemmas(text, language):
    """
    Extraction avec lemmatisation Stanza, fallback vers spaCy
    Utilise CLTK pour le grec si disponible
    """
    global _stanza_available, _cltk_available
    
    if language == "Greek" and _cltk_available:
        try:
            return _extract_lemmas_cltk_greek(text)
        except Exception as e:
            print(f"⚠️ Erreur CLTK pour grec: {e}, fallback vers Stanza")
            # Continue avec Stanza en fallback
    
    # Tentative Stanza en priorité
    if _stanza_available is None:  # Premier appel
        try:
            import stanza
            _stanza_available = True
            print("✅ Stanza disponible pour lemmatisation")
        except ImportError:
            _stanza_available = False
            print("⚠️ Stanza non disponible, utilisation de spaCy")
    
    # Utiliser Stanza si disponible
    if _stanza_available:
        try:
            return _extract_lemmas_stanza(text, language)
        except Exception as e:
            print(f"⚠️ Erreur Stanza pour {language}: {e}, fallback vers spaCy")
            return _extract_lemmas_spacy(text, language)
    
    # Fallback vers spaCy
    return _extract_lemmas_spacy(text, language)


def _extract_lemmas_stanza(text, language):
    """Extraction avec Stanza (méthode optimale)"""
    global _stanza_pipelines
    
    # Mapping des noms de langues vers codes Stanza
    stanza_language_map = {
        "English": "en",
        "French": "fr",
        "Italian": "it",
        "German": "de",
        "Spanish": "es",
        "Portuguese": "pt",
        "Latin": "la",          # Latin classique
        "Hebrew": "hbo",        # pour l'hébreu biblique
        "Greek": "grc"          # Grec ancien (si CLTK n'est pas disponible)
    }
    
    stanza_code = stanza_language_map.get(language)
    if not stanza_code:
        raise ValueError(f"Langue {language} non supportée par Stanza")
    
    # Charger le pipeline Stanza une seule fois par langue
    if stanza_code not in _stanza_pipelines:
        import stanza
        
        # Télécharger le modèle si nécessaire (silencieux)
        try:
            stanza.download(stanza_code, verbose=False)
        except:
            pass  # Le modèle existe peut-être déjà
        
        # Créer le pipeline avec lemmatisation
        _stanza_pipelines[stanza_code] = stanza.Pipeline(
            lang=stanza_code,
            processors='tokenize,pos,lemma',
            verbose=False
        )
        print(f"✅ Pipeline Stanza initialisé pour {language} ({stanza_code})")
    
    # Traiter le texte
    nlp = _stanza_pipelines[stanza_code]
    doc = nlp(text)
    lemmas = []
    
    for sentence in doc.sentences:
        for word in sentence.words:
            # Filtres adaptés selon la langue
            if stanza_code == "hbo":  # Hébreu biblique
                # Pour l'hébreu, vérifier les caractères hébraïques
                is_hebrew_char = any('\u0590' <= c <= '\u05FF' for c in word.text)
                if (is_hebrew_char and
                    word.upos in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']):
                    lemmas.append(word.lemma)  # Pas de .lower() pour préserver l'hébreu
            else:
                # Filtres standards pour les autres langues
                if (word.text.isalpha() and
                    word.upos in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']):
                    if word.upos == 'PROPN':
                        lemmas.append(word.lemma)  # Garder majuscule pour noms propres
                    else:
                        lemmas.append(word.lemma.lower())  # Minuscule pour le reste
    
    return lemmas


def _extract_lemmas_spacy(text, language):
    """Extraction avec spaCy (méthode fallback)"""
    global _spacy_pipelines
    
    # Charger le pipeline spaCy une seule fois par langue
    if language not in _spacy_pipelines:
        nlp = initializer(language)
        if nlp is None:
            raise ValueError(f"Langue {language} non supportée par spaCy")
        _spacy_pipelines[language] = nlp
        print(f"✅ Pipeline spaCy initialisé pour {language}")
    
    nlp = _spacy_pipelines[language]
    doc = nlp(text)
    lemmas = []
    
    # Traitement spécifique selon la langue
    if language == "Hebrew":
        # Pour l'hébreu, préserver la casse et filtrer les caractères hébreux
        for token in doc:
            is_hebrew_char = any('\u0590' <= c <= '\u05FF' for c in token.text)
            if (is_hebrew_char and
                token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']):
                lemmas.append(token.lemma_)  # Pas de .lower() pour préserver l'hébreu
                
    elif language == "Greek":
        # Pour le grec, traitement similaire au CLTK avec détection des noms propres
        proper_nouns_detected = set()
        
        # Détecter les noms propres potentiels (majuscule + PROPN)
        for token in doc:
            if (token.text[0].isupper() and
                token.pos_ == 'PROPN' and
                len(token.lemma_) > 3):
                proper_nouns_detected.add(token.lemma_)
        
        # Extraire les lemmes avec préservation de casse pour noms propres
        for token in doc:
            if (token.is_alpha and len(token.lemma_) >= 1 and
                not token.is_stop and
                token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'] and
                not token.lemma_ in ['καί', 'δέ', 'τε', 'γάρ', 'ἀλλά', 'οὖν', 'μέν', 'εἰ', 'ὁ', 'ἡ', 'τό', 'ὅτι', 'διά', 'Ὃς', 'μή', 'οὗτος']):
                
                # Garder la casse pour les noms propres détectés
                if (token.text[0].isupper() and
                    token.lemma_ in proper_nouns_detected and
                    len(token.lemma_) > 3):
                    lemmas.append(token.lemma_)  # Garder la casse originale
                else:
                    lemmas.append(token.lemma_.lower())
                    
    else:
        # Pour les autres langues (anglais, latin, etc.) avec préservation des noms propres
        for token in doc:
            if (token.is_alpha and len(token.lemma_) >= 3 and
                not token.is_stop and
                token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']):
                
                # Préserver la casse pour les noms propres
                if token.pos_ == 'PROPN' and token.text[0].isupper():
                    lemmas.append(token.lemma_)  # Garder casse pour noms propres
                else:
                    lemmas.append(token.lemma_.lower())  # Minuscule pour le reste
    
    return lemmas


def initializer(language):
    """
    Initialise spaCy pour une langue donnée (fonction de fallback)
    Renommée depuis spacy_intializer pour cohérence
    """
    try:
        if language == "English":
            return spacy.load("en_core_web_sm")
        elif language == "French":
            return spacy.load("fr_core_web_sm")
        elif language == "Italian":
            return spacy.load("it_core_web_sm")
        elif language == "Latin":
            return spacy.load("la_core_web_sm")
        elif language == "Spanish":
            return spacy.load("es_core_web_sm")
        elif language == "Portuguese":
            return spacy.load("pt_core_web_sm")
        elif language == "German":
            return spacy.load("de_core_web_sm")
        else:
            print(f"Langue {language} non prise en charge par spaCy")
            return None
    except OSError as e:
        print(f"Erreur chargement modèle spaCy pour {language}: {e}")
        return None

def _extract_lemmas_cltk_greek(text):
    """Extraction avec CLTK pour le grec ancien - API officielle v1.4.0"""
    global _cltk_greek_lemmatizer
    global _cltk_greek_proper_nouns
    
    # Initialiser le lemmatiseur une seule fois
    if _cltk_greek_lemmatizer is None:
        _cltk_greek_lemmatizer = GreekBackoffLemmatizer()
        print("✅ Lemmatiseur CLTK grec initialisé")
    
    # 1. Normaliser le texte (recommandé par CLTK)
    normalized_text = cltk_normalize(text)
    
    # 2. Tokeniser selon l'API officielle
    try:
        from cltk.tokenizers import GreekTokenizationProcess
        from cltk.core.data_types import Doc
        tokenizer = GreekTokenizationProcess()
        input_doc = Doc(raw=normalized_text)
        output_doc = tokenizer.run(input_doc=input_doc)
        words = output_doc.tokens
    except:
        # Fallback simple si problème avec tokenizer
        words = normalized_text.split()
    
    # 3. Lemmatiser avec CLTK
    lemma_results = _cltk_greek_lemmatizer.lemmatize(words)
    
    # 3. Détecter les noms propres avec NER CLTK
    proper_nouns_in_text = set()
    if _cltk_ner_available:
        try:
            ner_results = ner.tag_ner('grc', input_text=' '.join(words), output_type=list)
            for result in ner_results:
                if len(result) == 2 and result[1] == 'Entity':
                    proper_nouns_in_text.add(result[0])
        except Exception as e:
            print(f"⚠️ Erreur NER: {e}")

    # Collecter les noms propres détectés par NER avec un fichier text amélioré stocké en local pour corriger l'absence de lemmatisation de cet outil (déjà implémenté dans le cache de l'outil existant)
    for word, lemma in lemma_results:
        if (word in proper_nouns_in_text and word[0].isupper() and len(lemma) > 3):
            _cltk_greek_proper_nouns[lemma] = _cltk_greek_proper_nouns.get(lemma, 0) + 1
    
    # 4. Extraire les lemmes avec filtres
    lemmas = []
    for word, lemma in lemma_results:
        # Filtrer la ponctuation et les mots très courts
        if (len(lemma) >= 1 and
            not lemma.isdigit() and
            lemma not in [',', '.', ';', ':', '!', '?', '(', ')', '[', ']'] and
            not lemma in ['καί', 'δέ', 'τε', 'γάρ', 'ἀλλά', 'οὖν', 'μέν', 'εἰ', 'ὁ', 'ἡ', 'τό', 'ὅτι', 'διά', 'Ὃς', 'μή', 'οὗτος']):
            
            # Garder la casse pour les noms propres (commencent par majuscule) + NER
            if (word[0].isupper() and lemma in _cltk_greek_proper_nouns and len(lemma) > 3):
                lemmas.append(lemma)  # Garder la casse originale pour noms propres
            else:
                lemmas.append(lemma.lower())
    
    return lemmas
    
def reset_greek_proper_nouns():
    """Réinitialise le compteur de noms propres grecs pour un nouveau batch"""
    global _cltk_greek_proper_nouns
    _cltk_greek_proper_nouns = {}

import re
import random
import nltk
import json
from nltk.corpus import words, stopwords
from nltk.tokenize import sent_tokenize
import spacy
import pyinflect

# Download the necessary NLTK data if not already downloaded
# nltk.download('words')
# nltk.download('punkt')
# nltk.download('stopwords')


def load_spacy_model(model_name="en_core_web_sm"):
    """Load the spaCy model."""
    return spacy.load(model_name)

# Load the spaCy model once
nlp = load_spacy_model()

def create_typos_from_text(text, num_typos=5):
    def extract_words(text):
        return re.findall(r'\b\w+\b', text)

    def create_typo(word):
        if len(word) < 2:
            return word  # No typo possible for words with less than 2 characters
        idx = random.randint(0, len(word) - 2)
        typo_word = list(word)
        typo_word[idx], typo_word[idx + 1] = typo_word[idx + 1], typo_word[idx]
        return ''.join(typo_word)

    # Extract sentences from the text
    sentences = sent_tokenize(text)
    
    # Filter valid English words and exclude stopwords
    valid_words = set(words.words())
    stop_words = set(stopwords.words('english'))
    valid_words = valid_words - stop_words
    
    # Store the typos and their corresponding sentence numbers
    typo_info = []
    
    while len(typo_info) < num_typos:
        for i, sentence in enumerate(sentences):
            all_words = extract_words(sentence)
            valid_english_words = [word for word in all_words if word.lower() in valid_words]
            
            if valid_english_words:
                # Sample one word from the valid English words in the sentence
                sampled_word = random.choice(valid_english_words)
                typo = create_typo(sampled_word)
                
                # Replace only one occurrence of the sampled word with its typo
                sentence_with_typo = sentence.replace(sampled_word, typo, 1)
                sentences[i] = sentence_with_typo
                
                # Save the typo information
                typo_info.append({
                    "original": sampled_word,
                    "error": typo,
                    "sentence": i + 1  # Sentence number (1-based index)
                })
                
                if len(typo_info) >= num_typos:
                    break
    
    # Print the results
    #print("Original Text:\n", text)
    #print("\nText with Typos:\n", ' '.join(sentences))
    #print("\nTypos Information:\n", json.dumps(typo_info, indent=4))

    return typo_info


def replace_sampled_prepositions(text, sample_size=5):
    """
    Replace a sample of prepositions in the text with random prepositions to make the text wrong.
    
    Args:
        text (str): The input text.
        nlp (spacy.language.Language): The loaded spaCy model.
        sample_size (int): The number of prepositions to replace.
    
    Returns:
        str: A JSON string with the original and replaced prepositions.
    """
    # Define a list of prepositions to use as replacements
    prepositions_list = ['about', 'accross', 'after', 'against', 'along', 'among', 'around', 'as', 'at', 'before', 'behind', 'beside', 'between', 'beyond', 'by', 'despite', 'down', 'during', 'except', 'for', 'in', 'inside', 'into', 'near', 'of', 'on', 'onto', 'outside', 'over', 'past', 'since', 'through', 'to', 'toward', 'under', 'underneath', 'until', 'up', 'upon', 'with', 'within', 'without']
    
    # Process the text with spaCy
    doc = nlp(text)
    
    # Identify all prepositions and their positions in sentences
    prepositions = [(token.text, token.i, sent_idx) for sent_idx, sent in enumerate(doc.sents) for token in sent if token.pos_ == 'ADP']
    
    # Handle case where there are fewer prepositions than the sample size
    sample_size = min(sample_size, len(prepositions))
    
    # Sample a subset of prepositions to replace
    indices_to_replace = random.sample(range(len(prepositions)), sample_size)
    
    # Create replacements list
    replacements = []
    for idx in indices_to_replace:
        original_preposition, token_index, sentence_num = prepositions[idx]
        new_preposition = random.choice(prepositions_list)
        
        # Ensure the new preposition is different from the original
        while new_preposition == original_preposition:
            new_preposition = random.choice(prepositions_list)
        
        replacements.append({
            "original": original_preposition,
            "error": new_preposition,
            "sentence": sentence_num + 1  # Adjusting to 1-based index for sentences
        })
    
    return replacements #json.dumps(replacements, indent=2)

# Minimal irregular fallback map for very common verbs
IRREGULAR = {
    ("go", "VBZ"): "goes",
    ("go", "VBD"): "went",
    ("go", "VBN"): "gone",
    ("be", "VBZ"): "is",
    ("be", "VBP"): "are",
    ("be", "VBD"): "was",
    ("be", "VBN"): "been",
    ("have", "VBZ"): "has",
    ("have", "VBD"): "had",
    ("do", "VBZ"): "does",
    ("do", "VBD"): "did",
}

# Minimal irregular fallback map for very common verbs
IRREGULAR = {
    ("go", "VBZ"): "goes",
    ("go", "VBD"): "went",
    ("go", "VBN"): "gone",
    ("be", "VBZ"): "is",
    ("be", "VBP"): "are",
    ("be", "VBD"): "was",
    ("be", "VBN"): "been",
    ("have", "VBZ"): "has",
    ("have", "VBD"): "had",
    ("do", "VBZ"): "does",
    ("do", "VBD"): "did",
}

def inflect_form(lemma, target_tag):
    """
    Try pyinflect first; if that fails, fall back to IRREGULAR map
    then to naive suffix rules.
    """
    # pyinflect wants a spaCy token; make a dummy one
    dummy = nlp(lemma)[0]
    inflected = dummy._.inflect(target_tag)
    if inflected:
        return inflected
    
    # fallback to irregular dictionary
    if (lemma, target_tag) in IRREGULAR:
        return IRREGULAR[(lemma, target_tag)]
    
    # crude suffix fall-back (only for regular verbs)
    if target_tag == "VBZ":
        return lemma + "s"
    if target_tag == "VBD":
        return lemma + "ed"
    if target_tag == "VBP":
        return lemma                    # base already
    return None                         # give up

    
def break_sva(text, n_err=5, nlp=nlp):
    """
    Inject exactly `n_err` subject–verb agreement errors using correct inflections.
    Prints a warning if fewer than `n_err` errors are possible.
    """
    if isinstance(text, str):
        doc = nlp(text)
    else:
        doc = text

    # Collect all candidate (verb, sentence) pairs
    candidates = []
    for sent_no, sent in enumerate(doc.sents, start=1):
        for tok in sent:
            if tok.dep_ == "ROOT" and tok.pos_ == "VERB":
                if any(w.dep_ in ("nsubj", "nsubjpass") for w in tok.lefts):
                    candidates.append((tok, sent_no))

    random.shuffle(candidates)
    errors = []

    for tok, sent_no in candidates:
        if len(errors) >= n_err:
            break

        lemma = tok.lemma_.lower()
        tag   = tok.tag_

        if tag == "VBZ":
            target_tag = random.choice(["VBP", "VBD"])
        elif tag == "VBP":
            target_tag = random.choice(["VBZ", "VBD"])
        elif tag == "VBD":
            target_tag = random.choice(["VBZ", "VBP"])
        elif tag in ("VBG", "VBN"):
            target_tag = random.choice(["VBZ", "VBP"])
        else:
            continue

        wrong_form = inflect_form(lemma, target_tag)
        if not wrong_form:
            continue
        if wrong_form.lower() == tok.text.lower():
            continue

        errors.append({
            "original": tok.text,
            "error": wrong_form,
            "sentence": sent_no
        })

    # Fallback warning
    if len(errors) < n_err:
        print(f"[Warning] Only {len(errors)} errors could be generated (requested {n_err}).")

    return errors
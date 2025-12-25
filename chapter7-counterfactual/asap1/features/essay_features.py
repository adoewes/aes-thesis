from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
import numpy as np
import re
from nltk.corpus import stopwords, wordnet
from nltk import word_tokenize, sent_tokenize
from collections import Counter
from string import punctuation
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from spellchecker import SpellChecker
import string

import language_tool_python
tool = language_tool_python.LanguageTool('en-US')

# prompt = """More and more people use computers, but not everyone agrees that this benefits society. Those who support advances in technology believe that computers have a positive effect on people. They teach hand-eye coordination, give people the ability to learn about faraway places and people, and even allow people to talk online with other people. Others have different ideas. Some experts are concerned that people are spending too much time on their computers and less time exercising, enjoying nature, and interacting with family and friends. 

# Write a letter to your local newspaper in which you state your opinion on the effects computers have on people. Persuade the readers to agree with you.
# """

model = SentenceTransformer('bert-base-nli-mean-tokens')
prompt_sbert_file = 'sbert_prompt_asap1'
prompt_sbert =  joblib.load(prompt_sbert_file)

def preprocess_text(raw_text, remove_nonletters=False, remove_stopwords=False, stemming=False, return_list=False):

    # 1. Remove HTML
    text = BeautifulSoup(raw_text, 'lxml').get_text()

    # 2. Remove non-ASCII
    text_clean = re.sub(r"[^\x00-\x7F]+", " ", text)

    if remove_nonletters:
        text_clean = re.sub("[^a-zA-Z\.]", " ", text_clean)

    # 3. Convert to lower-case, split into words
    words = word_tokenize(text_clean.lower())

    # 4. Convert stopwords into Set (faster than List)
    # 5. Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        words = [w for w in words if not w in stop_words]

    # 6. Stemming
    if stemming:
        porter = PorterStemmer()
        stems = []
        for t in words:
            stems.append(porter.stem(t))
        words = stems

    if return_list:
        return words
    else:
        return(" ".join(words))


def get_sbert_features(essay):

    essay = preprocess_text(essay, True, False, False, False)
    
    sentence_list = sent_tokenize(essay)
    sentence_embeddings = model.encode(sentence_list)
    mean_sentence_embedding = np.average(sentence_embeddings, axis=0).reshape(1,-1)
    
    return mean_sentence_embedding


def get_prompt_relevance_sbert(essay, prompt_sbert_file):
    essay_sbert = get_sbert_features(essay)
    return cosine_similarity(essay_sbert.reshape(1,-1), prompt_sbert.reshape(1,-1))[0]


# Get the number of punctuations normalized by the number of sentences
def get_punctuation_density(text):
    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)
    
    count = 0
    total_sentences = len(sentences)
    
    for char in text:
        if char in string.punctuation:
            count += 1

    # print(count)
    # print(total_sentences)
    # Calculate normalized punctuation count using a ternary operator
    punctuation_density = count / total_sentences if total_sentences > 0 else 0
    
    return punctuation_density


def get_length_features(raw_text):

    essay = preprocess_text(raw_text, True, False, False, False)
    
    # 1. Answer Length (Character counts)
    answer_length = len(essay)
    
    # 2. Word count
    words = word_tokenize(essay)
    word_count = len(words)
    
    # 3. Average word length
    total_word_length = sum(len(word) for word in words)
    average_word_length = total_word_length / word_count if word_count > 0 else 0

    # 4. Unique words count
    unique_words_count = len(set(words))

    # 5. Punctuation density (normalized by number of sentences)
    # Create a regex pattern to keep letters and punctuation
    pattern = f"[^{string.ascii_letters}{punctuation}]"

    # Apply the regex to clean the text
    new_essay = re.sub(pattern, " ", raw_text)
    punctuation_density = get_punctuation_density(new_essay)
    
    #return answer_length, word_count, average_word_length, unique_words_count, punctuation_density
    
    return answer_length, word_count, average_word_length, unique_words_count, punctuation_density


def get_prompt_relevance(essay, prompt):
    # Preprocess both essay and prompt
    essay_tokens = preprocess_text(essay, remove_nonletters=True, remove_stopwords=True, stemming=False, return_list=True)
    prompt_tokens = preprocess_text(prompt, remove_nonletters=True, remove_stopwords=True, stemming=False, return_list=True)
    
    essay_tokens = [word for word in essay_tokens if word != '.']
    prompt_tokens = [word for word in prompt_tokens if word != '.']

    # Count overlapping words
    overlap_tokens = set(essay_tokens) & set(prompt_tokens)

    overlap_count = len(overlap_tokens)
    
    # Calculate relevance score
    essay_length = len(set(essay_tokens))
    prompt_length = len(set(prompt_tokens))
    
    relevance = overlap_count / max(essay_length, prompt_length)
    
    return relevance


def get_synonyms(word):
    synonyms = set()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            synonyms.add(lemma.name().lower())
    return synonyms

def get_prompt_relevance_synonym(essay, prompt):
    # Preprocess both essay and prompt
    essay_tokens = preprocess_text(essay, remove_nonletters=True, remove_stopwords=True, stemming=False, return_list=True)
    prompt_tokens = preprocess_text(prompt, remove_nonletters=True, remove_stopwords=True, stemming=False, return_list=True)

    essay_tokens = [word for word in essay_tokens if word != '.']
    prompt_tokens = [word for word in prompt_tokens if word != '.']
    
    # Calculate synonyms for each token
    essay_synonyms = []
    for token in essay_tokens:
        essay_synonyms.extend(get_synonyms(token))
    
    prompt_synonyms = []
    for token in prompt_tokens:
        prompt_synonyms.extend(get_synonyms(token))

    # Get overlapping tokens
    overlapping_tokens = set(essay_tokens + essay_synonyms) & set(prompt_tokens + prompt_synonyms)
    # Count overlapping words (including synonyms)
    overlap_count = len(overlapping_tokens)
    
    # Calculate relevance score
    essay_length = len(set(essay_tokens + essay_synonyms))
    prompt_length = len(set(prompt_tokens + prompt_synonyms))
    
    relevance = overlap_count / max(essay_length, prompt_length)
    
    return relevance


# Get the number of spelling errors (word) normalized by the number of words
def get_spelling_errors_density_count(essay):
    essay = preprocess_text(essay, True, False, False, True)

    essay = [word for word in essay if word != '.']

    spell = SpellChecker()

    spell_errors = spell.unknown(essay)
    
    spell_error_count = len(spell_errors)

    spell_error_density = spell_error_count / len(essay) if len(essay) > 0 else 0

    return spell_error_density, spell_error_count


# get language errors (including spelling errors), the result will be reduced by the number of spelling errors from the other function 
def get_language_errors_density(essay, spell_error_count):
    matches = tool.check(essay)
    language_errors = len(matches) - spell_error_count

    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(essay)
    total_sentences = len(sentences)

    language_errors_density = language_errors / total_sentences if total_sentences > 0 else 0
    
    return language_errors_density


def extract_essay_features(essay, prompt):
    length_features = get_length_features(essay)

    length_features = np.array(length_features).reshape(1,-1)
    
    prompt_relevance = get_prompt_relevance(essay, prompt)
    prompt_relevance = np.array(prompt_relevance).reshape(1,-1)

    prompt_relevance_synonym = get_prompt_relevance_synonym(essay, prompt)
    prompt_relevance_synonym = np.array(prompt_relevance_synonym).reshape(1,-1)

    prompt_sbert_file = 'sbert_prompt_asap1'
    
    prompt_relevance_sbert = get_prompt_relevance_sbert(essay, prompt_sbert_file)
    prompt_relevance_sbert = np.array(prompt_relevance_sbert).reshape(1,-1)

    spell_error_density, spell_error_count = get_spelling_errors_density_count(essay)
    spell_error_density = np.array(spell_error_density).reshape(1,-1)

    language_error_density = get_language_errors_density(essay, spell_error_count)
    language_error_density = np.array(language_error_density).reshape(1,-1)

    sbert_features = get_sbert_features(essay)

    essay_features = np.concatenate((length_features,
                                    prompt_relevance, prompt_relevance_synonym, prompt_relevance_sbert,
                                    spell_error_density, language_error_density,
                                    sbert_features), axis=1)
    
    #print(essay_features.shape)
    
    return essay_features


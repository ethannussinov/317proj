import fitz #PyMuPDF, python package for efficient text extraction from PDFs

import re
import requests
import numpy as np

import nltk
from nltk import pos_tag

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import sentencepiece
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import GPT2Tokenizer, GPT2LMHeadModel, set_seed


from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

print('hello')


def read_pdf(file_path):
    '''
    This function takes an argument of the file path of a linguistic paper in PDF format and returns its text content.
    '''
    text_content = ''
    
    try:
        # open the PDF file
        with fitz.open(file_path) as pdf:
            # iterate over each page
            for page in pdf:
                # extract text from the page and add it to the content
                text_content += page.get_text()
    # error handling            
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    return text_content


def preprocess_text(text_content):
    '''
    This function takes a text argument and returns the text preprocessed, without references or acknowledgements, and with normalized text.
    '''
    # remove references section
    text_content = re.sub(r'\n?References\n?.*', '', text_content, flags=re.DOTALL)
    # remove any lines that contain URLs, DOI, or email addresses
    text_content = re.sub(r'\S*@\S*\s?', '', text_content)
    text_content = re.sub(r'http\S+', '', text_content)
    text_content = re.sub(r'doi:\S+', '', text_content)
    # normalize text by converting to lowercase and removing special characters
    # keep hyphenated terms and apostrophes within words
    text_content = re.sub(r'[^a-zA-Z0-9\s\-\']', ' ', text_content) #will need to refine, causing an issue ill point out in a minute
    text_content = text_content.lower()
    
    # remove any double or more spaces with a single space
    text_content = re.sub(' +', ' ', text_content)
    
    # strip whitespace at the beginning and end of the text
    text_content = text_content.strip()

    return text_content



def process_text(text):
    '''
    This function takes a text argument and returns the text tokenized and POS tagged, utilizing NLTK models.

    '''
    # tokenization
    tokens = word_tokenize(text)
    
    # POS tagging
    tagged_tokens = pos_tag(tokens)
    
    return tagged_tokens



def extract_terms_with_tfidf(preprocessed_text, top_n=50):
    """
    This function takes preprocessed text and returns the top_n terms ranked by their TF-IDF score,
    excluding purely numerical terms and additional common words.
    """
    # initialize tfidf vectorizer with a token pattern that excludes pure numbers, initial issue i was encountering
    vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'(?u)\b[A-Za-z]+\b')

    # apply the vectorizer to the preprocessed text to create the tfidf matrix
    tfidf_matrix = vectorizer.fit_transform([preprocessed_text])

    # get the names of the features/terms
    feature_names = vectorizer.get_feature_names_out()

    # sum the tfidf scores for each term to get an overall score
    sum_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()

    # sprt by overall tfidf score in descending order
    sorted_indices = sum_scores.argsort()[::-1]
    sorted_scores = sum_scores[sorted_indices]

    # extract highest score terms
    top_terms_with_scores = [(feature_names[idx], sorted_scores[idx]) for idx in sorted_indices if sorted_scores[idx] > 0]

    # exclude domain specific words (growing list, ineffective method)
    domain_stopwords = set(['study', 'research', 'data', 'analysis', 'result', 'significant', 'language', 'word', 'case', 'type', 'phonology', 'linguistics', 'paper', 'subject', 'method', 'discussion', 'conclusion', 'abstract', 'introduction', 'section', 'figure', 'table', 'appendix'])
    tfidf_threshold = 0.05  # adjustable based on results

    # apply filters
    filtered_terms = [(term, score) for term, score in top_terms_with_scores if not re.fullmatch(r'\d+', term) and term not in domain_stopwords and score > tfidf_threshold]

    # extract top_n terms if top_n is defined, otherwise return all filtered terms
    if top_n:
        filtered_terms = filtered_terms[:top_n]

    refined_terms = [term for term, score in filtered_terms]

    return refined_terms




def enhanced_term_identification(tagged_tokens, linguistic_terms_set=None):
    '''
    This function integrates various strategies to identify and filter terms for a glossary.
    It incorporates Noun Phrase Extraction, Frequency Filtering, Stopword Removal,
    Linguistic Term Filter, Named Entity Recognition (NER), and Heuristic Rules.

    Parameters:
    - tagged_tokens: a list of POS-tagged tokens from the text.
    - linguistic_terms_set: an optional set of known linguistic terms for additional filtering.

    Returns:
    - A dictionary of filtered terms and their frequencies.
    '''
    stop_words = set(stopwords.words('english'))
    # initialize a list to store noun phrases
    noun_phrases = []

    # Noun Phrase Extraction
    grammar = "NP: {<DT>?<JJ>*<NN|NNS>+}"
    cp = nltk.RegexpParser(grammar)
    tree = cp.parse(tagged_tokens)

    for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
        # exclude stopwords and single characters from noun phrases
        phrase = " ".join(word for word, tag in subtree.leaves() if word.lower() not in stop_words and len(word) > 1)
        if phrase:
            noun_phrases.append(phrase)

    # frequency filtering
    term_counts = Counter(noun_phrases) #somewhat counterproductive

    # optional: filter by predefined linguistic terms
    if linguistic_terms_set:
        term_counts = {term: count for term, count in term_counts.items() if term in linguistic_terms_set}

    # filter terms by their frequency and length (for example, appearing more than once) 'heuristics'
    filtered_terms = {term: count for term, count in term_counts.items() if count > 1 and len(term) > 3}

    return filtered_terms


def generate_definition_gpt2(term):
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Ensure the pad token is set for padding purposes
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Adjust tokenizer settings for left padding
    tokenizer.padding_side = 'left'

    # Simplified and direct prompt
    prompt = f"Define the term '{term} in the context of an academic linguistic paper"
    inputs = tokenizer(prompt, return_tensors='pt', padding='max_length', max_length=150)

    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=200,
        max_new_tokens=70,  # Adjusting to allow more room for a complete definition
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    definition = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return definition


term = "vowel deletion"
definition = generate_definition_gpt2(term)
print("Generated Definition:", definition)
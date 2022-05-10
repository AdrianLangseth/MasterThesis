import tensorflow as tf
import gensim
import pandas as pd
from tokenizer_embedder import tokenize_article
from tqdm import tqdm
from nltk import word_tokenize, sent_tokenize


def generate_corpus(corpus_text:pd.Series) -> list:
    if pd.isna(corpus_text):
        print('Found NaN\'s, removing ...')
        corpus_text[corpus_text.isna()] = ''

    assert sum(pd.isna(corpus_text)) == 0

    corpus = []
    for article in tqdm(corpus_text, desc='Iterating over Articles'):
        for sent in sent_tokenize(article):
            corpus.append(word_tokenize(sent, language='norwegian'))

    return corpus


def generate_embedder(corpus:list):
    """
    Generates a W2V embedder from given corpus
    :param corpus: list of lists of tokenized sentences.
    :return: a gensim W2V model trained on the given corpus.
    """
    return gensim.models.Word2Vec(tqdm(corpus))


def save_embedder(model:gensim.models.Word2Vec, path:str='data/embedder_model') -> None:
    model.save(path)


def load_embedder(path:str='data/embedder_model') -> gensim.models.Word2Vec:
    return gensim.models.Word2Vec.load(path)
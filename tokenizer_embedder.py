import gensim
import nltk
import numpy as np
import pandas as pd
import stopwordsiso
import tensorflow as tf
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm

# TODO:
# - Somethings fucky with articles with length 76. They have body as "P U B L I S E R T : 2 1 M A R S" OSV.
#   - Need to remove these from dataset.
#   - df['body_length'] = df['body'].apply(lambda x: len(x.split(' ')) if pd.notna(x) else 0)
#   - same for 68, somewhat for 82 as well
#   - not only for
###################

#  ########### TOKENIZATION #######################
import globals


def tokenize_sentence(text_sentence: str) -> list:
    text_sentence = text_sentence.lower()
    tokens = nltk.word_tokenize(text_sentence, language='norwegian')
    if (len(tokens) >= 3) and (tokens != ['saken', 'oppdateres', '.']):
        if (tokens[:2] == ['publisert', ':']) and (len(tokens) > 13):
            tokens = tokens[13:]
        return tokens
    return []


def tokenize_article(body: str, max_length=1000):
    words = []
    if pd.isna(body):
        return words
    body = body.replace('|', '.')
    sentence_count = 0
    for sentence in nltk.sent_tokenize(body, language='norwegian'):
        tokens = tokenize_sentence(sentence)
        if tokens:
            words.extend(tokens)
            sentence_count += 1
            if len(words) >= max_length:
                break
    return words[:max_length]


# TODO: Remove max sentence size
# Todo: fix problems with "P U B L I S E R T 2 0 : 3 4 ..."
def tokenize_norwegian_article(text, max_words=None, max_sentences=None):
    # Removing pipes for correct sentence tokenization
    text = text.replace('|', '.')
    words_tokenized = []
    sent_count = 0
    for sentence in nltk.tokenize.sent_tokenize(text, language='norwegian'):
        sent_tokenized = nltk.tokenize.word_tokenize(sentence, language='norwegian')
        if len(sent_tokenized) >= 3 and \
                sent_tokenized != ['Saken', 'oppdateres', '.']:  # and sent_tokenized[-1] in ['.', '!', '?', ';']
            sent_count += 1
            words_tokenized.extend(sent_tokenized)
            if sent_count == max_sentences:
                break
    return words_tokenized[:max_words]


def tokenize_dataset(df: pd.DataFrame):
    tqdm.pandas(desc='Tokenizing Dataset')

    df.title = df.title.progress_apply(tokenize_norwegian_article,
                                       args=(None, None))  # config.MAX_TITLE_WORD_TOKENS, config.MAX_TITLE_SENTENCE_TOKENS))
    df.body = df.body.progress_apply(tokenize_norwegian_article,
                                     args=(None, None))  # (config.MAX_ARTICLE_WORD_TOKENS, config.MAX_ARTICLE_SENTENCE_TOKENS))

    df.title = df.title.apply(lambda x: [i for i in x if len(i) > 1])  # Removes 1 character words and punctuation
    df.body = df.body.apply(lambda x: [i for i in x if len(i) > 1])  # Removes 1 character words and punctuation


def tokenize_several(text_series: pd.Series):
    x = []
    for article in tqdm(text_series):
        x.append(tokenize_article(article))
    return x


#  ####################### TOKENIZATION #######################


#  ####################### CORPUS #######################
def generate_corpus(df: pd.DataFrame) -> list:
    titles = df.title
    bodies = df.body

    corpus_text = pd.concat([titles, bodies], ignore_index=True)
    stopwords = set([i.translate(i.maketrans('öû', 'æø')) for i in stopwordsiso.stopwords('no')]).union(['saken', 'oppdateres'])

    if sum(corpus_text.isna()):
        print('Found NaN\'s, removing ...')
        corpus_text[corpus_text.isna()] = ''

    assert sum(pd.isna(corpus_text)) == 0

    corpus = []
    for textual_entity in tqdm(corpus_text, desc='Generating corpus for embedder'):
        for sent in sent_tokenize(textual_entity):
            l = [i for i in word_tokenize(sent, language='norwegian') if ((len(i) > 1) and (i.lower() not in stopwords))]
            if len(l):
                corpus.append(l)

    return corpus


#  ####################### CORPUS #######################

#  ####################### Text Vectorize #######################

def vectorize_textual_data(data: pd.DataFrame, embedder):
    tqdm.pandas(desc='Vectorizing Dataset')
    title = data.title.progress_apply(
        lambda x: [embedder.wv.key_to_index[i] for i in x if i in embedder.wv.key_to_index.keys()])
    body = data.body.progress_apply(
        lambda x: [embedder.wv.key_to_index[i] for i in x if i in embedder.wv.key_to_index.keys()])
    print(f'----- MIN_COUNT: {globals.data_params["min_count"]} --------')
    print(f'Title = []: {(title.apply(len)==0).mean()}')
    print(f'Vocab_size: {embedder.cum_table.shape[0]}')
    print(f'Body = []: {(body.apply(len)==0).mean()}')
    print(f'Tile & Body = []: {(((data.title.apply(len) + data.body.apply(len))==0).mean())}')
    print('-----------------------------')

    return title, body


#  ####################### Text Vectorize #######################

#  ####################### StopWords & Punctuation #######################

# Removed. Stopword and punctuation removal is done in CORPUS

#  ####################### StopWords & Punctuation #######################

#  ####################### EMBEDDER #######################
def load_embedder(path: str = 'data/embedder/embedder_model') -> gensim.models.Word2Vec:
    return gensim.models.Word2Vec.load(path)


def build_embedder(corpus: list):
    """
    Generates a W2V embedder from given corpus
    :param corpus: list of lists of tokenized sentences.
    :return: a gensim W2V model trained on the given corpus.
    """
    return gensim.models.Word2Vec(tqdm(corpus, desc="Building embedder"),
                                  vector_size=globals.data_params['vector_size'],
                                  window=globals.data_params['window'],
                                  min_count=globals.data_params['min_count'],
                                  # max_vocab_size=config.embedder_params['max_vocab_size'],
                                  sorted_vocab=1,
                                  )


def save_embedder(model: gensim.models.Word2Vec, path: str = 'data/embedder/embedder_model') -> bool:
    model.save(path)
    return True


def get_embedding_layer_from_gensim(path: str = './data/embedder/embedder_model.wv.vectors.npy'):
    wv = np.load(path)[:globals.data_params['vocabulary_size'], :]
    (vocab_size, embedding_dim) = wv.shape

    embedded_sequences = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        embeddings_initializer=tf.keras.initializers.Constant(wv),
        trainable=False, )

    return embedded_sequences
#  ########### EMBEDDER #######################

import argparse
import pandas as pd
import os
from collections import defaultdict, Counter
import json
import numpy as np
import nltk
from dateutil.parser import parse
from joblib import Parallel, delayed
from sklearn.utils import class_weight
import tensorflow as tf
from tqdm import tqdm
import pickle
from time import time

from build_embedder import generate_corpus
from tokenizer_embedder import build_embedder

# GLOBAL VARIABLES
CATEGORIES_TO_IGNORE = ['bolig', 'abonnement']
SITES_TO_IGNORE = ['kundeservice.adressa.no']
PAD_TOKEN = '<PAD>'
UNFREQ_TOKEN = "<UNF>"


# ## ARTICLE PARSING START
def load_contents_from_folder(path, subset=False):
    if subset:
        files_list = sorted(os.listdir(path))[0:2000]
    else:
        files_list = sorted(os.listdir(path))

    articles = load_contents_from_files_list(path, files_list)

    # Merging articles in a data frame
    news_df = pd.DataFrame([article for article in articles])

    # Filtering out news with invalid categories or from specific sites
    news_df = news_df[(~news_df['category0'].isin(CATEGORIES_TO_IGNORE)) & \
                      (~news_df['site'].astype(str).isin(SITES_TO_IGNORE))]

    news_df.drop_duplicates(subset='id', keep='first', inplace=True)
    return news_df


def load_contents_from_files_list(root_path, files_list):
    total_contents = 0
    invalid_contents_count = 0
    articles = []

    for idx, filename in enumerate(tqdm(files_list)):
        file_content = parse_content_file(os.path.join(root_path, filename))
        if file_content == None:
            invalid_contents_count += 1
        else:
            articles.append(file_content)
        total_contents += 1

    print("Processed content files: {} - Empty files: {} - Valid articles: {}".format(total_contents,
                                                                                      invalid_contents_count,
                                                                                      len(articles)))
    return articles


def parse_content_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as fi:
        try:
            for line in fi:
                if line.strip() == 'null':
                    return None
                content = parse_content(line)
                # Returns only the first json from content file, as others are identical, only "score" is different
                return content
        except Exception as e:
            print('Error processing file "{}": {}'.format(file_path, e))
            raise e


def parse_content(line):
    content_json = parse_content_general(line)
    content_raw = defaultdict(str, content_json)

    publishtime = content_raw['publishtime'] if content_raw['publishtime'] != '' else content_raw['createtime']
    # Converting to unix timestamp in miliseconds
    publishtime_ts = int(parse(publishtime).timestamp()) * 1000

    # author_1st = content_raw['author'][0] if type(content_raw['author']) == list else content_raw['author']

    # if type(content_raw['heading']) == list:
    #     heading = set(content_raw['heading'])  # Set to remove repeated phrases
    # else:
    #     heading = [content_raw['heading']]

    ## NaN checks
    if pd.isna(content_raw['title']):
        content_raw['title'] = ''
    # if pd.isna(content_raw['teaser']):
    #     content_raw['teaser'] = ''
    # if pd.isna(content_raw['heading']):
    #     content_raw['heading'] = ''
    if pd.isna(content_raw['body']):
        content_raw['body'] = ''

    new_content = {'id': content_raw['id'],
                   'url': content_raw['url'],
                   'site': unique_list_if_str(content_raw['og-site-name'])[0],
                   #'adressa-access': content_raw['adressa-access'],  # (free, subscriber)
                   #'author_1st': author_1st if author_1st != '' else '',  # 3777 unique
                   'publishtime': publishtime,
                   'created_at_ts': publishtime_ts,

                   'title': content_raw['title'],
                   #'teaser': content_raw['teaser'],
                   #'heading': content_raw['heading'],
                   'body': content_raw['body'],

                   # Extracted using NLP techniques (by Adressa)
                   #'concepts': ','.join(unique_list_if_str(content_raw['kw-concept'])),  # 98895 unique
                   #'entities': ','.join(unique_list_if_str(content_raw['kw-entity'])),  # 150214 unique
                   'locations': ','.join(unique_list_if_str(content_raw['kw-location'])),  # 5533 unique
                   # 'persons': ','.join(unique_list_if_str(content_raw['kw-person'])),  # 53535 unique

                   # Categories and keywords tagged by the journalists of Adresseavisen and may be of variable quality (label)
                   'category0': content_raw['category0'],  # 39 unique
                   'category1': content_raw['category1'] if 'category1' in content_raw else '',  # 126 unique
                   # 'category2': content_raw['category2'] if 'category2' in content_raw else '',  # 75 unique
                   # 'keywords': content_raw['keywords'],  # 6489 unique
                   }

    return new_content


def parse_content_general(line):
    content_raw = json.loads(line)

    new_content = {}
    for key in content_raw:
        if key == 'fields':
            for field in content_raw['fields']:
                value = field['value']
                if field['field'] == 'body':
                    value = ' '.join(value)
                new_content[field['field']] = value
        else:
            new_content[key] = content_raw[key]

    return new_content


def unique_list_if_str(value):
    if type(value) == list:
        return value
    else:
        return [value]



def preprocess_article_data(loaded_data):
    pass


def process_naml_cat_features(news_df):

    relevant_cols = ['id', 'url', 'publishtime', 'title',
                     'body', 'locations', 'category0', 'category1']

    # ID
    article_id_encoder = get_categ_encoder_from_values(news_df['id'])
    news_df['id_encoded'] = transform_categorical_column(news_df['id'], article_id_encoder)

    # Cat 0
    category0_encoder = get_categ_encoder_from_values(news_df['category0'].unique())
    news_df['category0_encoded'] = transform_categorical_column(news_df['category0'], category0_encoder)
    category0_class_weights = class_weight.compute_class_weight('balanced',
                                                                classes=news_df['category0_encoded'].unique(),
                                                                y=news_df['category0_encoded'])

    # Cat 1
    category1_encoder = get_categ_encoder_from_values(news_df['category1'].unique())
    news_df['category1_encoded'] = transform_categorical_column(news_df['category1'], category1_encoder)
    category1_class_weights = class_weight.compute_class_weight('balanced',
                                                                classes=news_df['category1_encoded'].unique(),
                                                                y=news_df['category1_encoded'])

    # Location
    news_df['locations'] = news_df['locations'].apply(lambda x: comma_sep_values_to_list(x) if pd.notna(x) else [])

    locations_encoder = get_encoder_from_freq_values_in_list_column(news_df['locations'])
    news_df['locations_encoded'] = transform_categorical_list_column(news_df['locations'], locations_encoder)

    loc_flattened = np.concatenate(news_df['locations_encoded'], dtype=int, casting='unsafe')

    location_class_weights = class_weight.compute_class_weight('balanced',
                                                               classes=np.unique(loc_flattened),
                                                               y=loc_flattened)

    print('Articles - unique count {}'.format(len(article_id_encoder)))
    print('Category0 - unique count {}'.format(len(category0_encoder)))
    print('Category1 - unique count {}'.format(len(category1_encoder)))
    print('Locations - freq. unique count {}'.format(len(locations_encoder)))

    cat_features_encoders = {'article_id': article_id_encoder,
                             'category0': category0_encoder,
                             'category1': category1_encoder,
                             'locations': locations_encoder,
                             }

    labels_class_weights = {
        'category0': category0_class_weights,
        'category1': category1_class_weights,
        'locations': location_class_weights
    }

    return cat_features_encoders, labels_class_weights


def flatten_list_series(series_of_lists):
    return pd.DataFrame(series_of_lists.apply(pd.Series).stack().reset_index(name='item'))['item']


def get_freq_values(series, min_freq=100):
    flatten_values_counts = series.groupby(series).size()
    return flatten_values_counts[flatten_values_counts >= min_freq].sort_values(ascending=False).reset_index(
        name='count')


def get_freq_values_series_of_lists(series_of_lists, min_freq):
    flatten_values = flatten_list_series(series_of_lists)
    flatten_values_counts = get_freq_values(flatten_values, min_freq)
    return flatten_values_counts


def get_encoder_from_freq_values(series, min_freq=100):
    freq_values_counts_df = get_freq_values(series, min_freq=min_freq)
    encoder = get_categ_encoder_from_values(freq_values_counts_df[freq_values_counts_df.columns[0]].unique(),
                                            include_unfrequent_token=True)
    return encoder


def transform_categorical_column(series, encoder):
    return series.apply(lambda x: encode_categ_feature(x, encoder))


def get_encoder_from_freq_values_in_list_column(series, min_freq=100):
    freq_values_counts_df = get_freq_values_series_of_lists(series, min_freq=min_freq)
    encoder = get_categ_encoder_from_values(freq_values_counts_df[freq_values_counts_df.columns[0]].unique(),
                                            include_unfrequent_token=False)
    return encoder


def transform_categorical_list_column(series, encoder):
    return series.apply(lambda l: list([encoder[val] for val in l if val in encoder]))


def comma_sep_values_to_list(value):
    return list([y.strip() for y in value.split(',') if y.strip() != ''])


def get_sample_weight_inv_freq(class_value, classes_count, numerator=10.0):
    return numerator / classes_count[class_value]


# ## ARTICLE PARSING END

# ## UTILS START

def serialize(filename, obj):
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle)


def deserialize(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)


def get_categ_encoder_from_values(values, include_pad_token=True, include_unfrequent_token=False):
    encoder_values = []
    if include_pad_token:
        encoder_values.append(PAD_TOKEN)
    if include_unfrequent_token:
        encoder_values.append(UNFREQ_TOKEN)
    encoder_values.extend(values)
    encoder_ids = list(range(len(encoder_values)))
    encoder_dict = dict(zip(encoder_values, encoder_ids))
    return encoder_dict


def encode_categ_feature(value, encoder_dict):
    if value in encoder_dict:
        return encoder_dict[value]
    else:
        return encoder_dict[UNFREQ_TOKEN]


def save_article_cat_encoders(output_path, cat_features_encoders, labels_class_weights):
    to_serialize = (cat_features_encoders, labels_class_weights)
    serialize(output_path, to_serialize)

def load_article_cat_encoders(load_path:str):
    cat_features_encoders, labels_class_weights = deserialize(load_path)
    return cat_features_encoders, labels_class_weights

# ## UTILS END

def main():
    content_folder = 'home/lemeiz/content_refine'
    encoder_save_fp = 'data/encoders/encoders_pickle'

    df = load_contents_from_folder(content_folder, subset=False)
    df.to_csv('data/raw_df.csv')
    short_df = df[
        ['id', 'url', 'publishtime', 'title', 'teaser', 'heading', 'body', 'locations', 'category0', 'category1']]
    short_df.to_csv('data/cropped_df.csv')
    # short_df = pd.read_csv('data/cropped_df.csv')
    #cat_features_encoders, labels_class_weights = process_naml_cat_features(short_df)
    #save_article_cat_encoders(encoder_save_fp, cat_features_encoders, labels_class_weights)
    #short_df.to_csv(output_content_csv_path, index=False)


def execute_full_data_preperation(content_folder:str = 'home/lemeiz/content_refine'):
    # Load_data
    loaded_data = load_contents_from_folder(content_folder, subset=True)
    # Preprocess data
    preprocessed_data = preprocess_article_data(loaded_data)
    # generate categorical encodings and label weights
    cat_features_encoders, labels_class_weights = process_naml_cat_features(preprocessed_data)
    # Tokenize all data
    textuals = []

    corpus_text = None
    # Build embedder
    corpus = generate_corpus(corpus_text=preprocessed_data.body)  # Generate corpus
    corpus.extend(generate_corpus(corpus_text=preprocessed_data.title))  # add corpus w
    emb = build_embedder(corpus)
    # save preprocessed data, encoders, embedder.

    emb.save('data/embedder_model')



if __name__ == '__main__':
    main()


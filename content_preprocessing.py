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
import warnings

from tokenizer_embedder import build_embedder, tokenize_dataset, save_embedder, generate_corpus, vectorize_textual_data

warnings.filterwarnings("ignore")

# GLOBAL VARIABLES
CATEGORIES_TO_IGNORE = ['bolig', 'abonnement']
SITES_TO_IGNORE = ['kundeservice.adressa.no']
PAD_TOKEN = '<PAD>'
UNFREQ_TOKEN = "<UNF>"


# ## ARTICLE PARSING START
def parse_content_general(line):
    """
    Takes in a json line and returns the raw data in dictionary-form.
    :param line: line of json file
    :return: dict of json data
    """
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


def parse_content(line):
    """
    Parses the content. Takes in a line, gives out a dictionary of the parsed content.
    """
    content_json = parse_content_general(line)
    content_raw = defaultdict(str, content_json)

    publishtime = content_raw['publishtime'] if content_raw['publishtime'] != '' else content_raw['createtime']
    # Converting to unix timestamp in miliseconds
    publishtime_ts = int(parse(publishtime).timestamp()) * 1000

    # todo: move all to preprocess?
    new_content = {'id': content_raw['id'],
                   'url': content_raw['url'],
                   'site': unique_list_if_str(content_raw['og-site-name'])[0],
                   'adressa-access': content_raw['adressa-access'],  # (free, subscriber)
                   'publishtime': publishtime,
                   'created_at_ts': publishtime_ts,

                   # Content
                   'title': content_raw['title'],
                   'body': content_raw['body'],

                   # Extracted using NLP techniques (by Adressa)
                   'locations': content_raw['kw-location'],  # 5533 unique

                   # Categories and keywords tagged by the journalists
                   'category0': content_raw['category0'],  # 39 unique
                   'category1': content_raw['category1'] if 'category1' in content_raw else ''  # 126 unique
                   }

    return new_content



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


def load_contents_from_files_list(root_path, files_list):
    total_contents = 0
    invalid_contents_count = 0
    articles = []

    "Parsing articles:"
    for idx, filename in enumerate(tqdm(files_list)):
        file_content = parse_content_file(os.path.join(root_path, filename))
        if file_content is None:
            invalid_contents_count += 1
        else:
            articles.append(file_content)
        total_contents += 1

    print(
        f"Processed content files: {total_contents} - "
        f"Empty files: {invalid_contents_count} - "
        f"Valid articles: {len(articles)}")

    return articles


def load_contents_from_folder(path, subset=False):
    if subset:
        files_list = sorted(os.listdir(path))[0:2000]
    else:
        files_list = sorted(os.listdir(path))

    articles = load_contents_from_files_list(path, files_list)

    # Merging articles in a data frame
    news_df = pd.DataFrame([article for article in articles])
    return news_df


def unique_list_if_str(value):
    if type(value) == list:
        return value
    else:
        return [value]


def preprocess_article_data(loaded_data: pd.DataFrame) -> pd.DataFrame:
    # Filtering out news with invalid categories or from specific sites
    data = loaded_data[(~loaded_data['category0'].isin(CATEGORIES_TO_IGNORE)) & \
                       (~loaded_data['site'].astype(str).isin(SITES_TO_IGNORE))]

    data.drop_duplicates(subset='id', keep='first', inplace=True)

    data[['body', 'title']] = data[['body', 'title']].replace(np.nan, '')
    data.title = data.title.apply(lambda x: x.lower())
    data.body = data.body.apply(lambda x: x.lower())

    data['locations'] = data['locations'].apply(lambda x: [] if x == '' else x)
    data['locations'] = data['locations'].apply(lambda x: [x] if type(x) is str else x)

    return data


def process_naml_cat_features(df: pd.DataFrame) -> (pd.DataFrame, dict, dict):
    news_df = df.copy()

    # ID
    article_id_encoder = get_categ_encoder_from_values(news_df['id'])
    news_df['id_encoded'] = transform_categorical_column(news_df['id'], article_id_encoder)

    news_df.set_index('id_encoded', inplace=True)

    # Category
    category0_encoder = get_categ_encoder_from_values(news_df['category0'].unique())
    news_df['category0_encoded'] = transform_categorical_column(news_df['category0'], category0_encoder)
    category0_class_weights = class_weight.compute_class_weight('balanced',
                                                                classes=news_df['category0_encoded'].unique(),
                                                                y=news_df['category0_encoded'])

    # Subcategory
    category1_encoder = get_categ_encoder_from_values(news_df['category1'].unique())
    news_df['category1_encoded'] = transform_categorical_column(news_df['category1'], category1_encoder)
    category1_class_weights = class_weight.compute_class_weight('balanced',
                                                                classes=news_df['category1_encoded'].unique(),
                                                                y=news_df['category1_encoded'])
    # Location
    locations_encoder = get_encoder_from_freq_values_in_list_column(news_df['locations'])
    news_df['locations_encoded'] = transform_categorical_list_column(news_df['locations'], locations_encoder)

    loc_flattened = []
    for i in news_df['locations_encoded']:
        loc_flattened.extend(i)

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

    return news_df, cat_features_encoders, labels_class_weights


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


def load_article_cat_encoders(load_path: str):
    cat_features_encoders, labels_class_weights = deserialize(load_path)
    return cat_features_encoders, labels_class_weights


# ## UTILS END


def execute_full_data_preperation(
        content_folder: str = 'home/lemeiz/content_refine',
        out_fp: str = 'data/clean_encoded_data.csv',
        encoders_pickle_path: str = 'data/encoders_pickle',
        embedder_out_fp: str = 'data/embedder/embedder_model'
):
    # Load_data
    loaded_data = load_contents_from_folder(content_folder, subset=True)
    # Preprocess data
    preprocessed_data: pd.DataFrame = preprocess_article_data(loaded_data)
    # generate categorical encodings and label weights
    data, cat_features_encoders, labels_class_weights = process_naml_cat_features(preprocessed_data)
    # Generate corpus from sentences in data.
    corpus = generate_corpus(data)  # Generate corpus
    # Tokenize all data
    tokenize_dataset(data)
    # Build embedder
    emb = build_embedder(corpus)
    # Encode textual data
    data.title, data.body = vectorize_textual_data(data, emb)
    # save preprocessed data, encoders, embedder.
    data.to_csv(out_fp)
    save_article_cat_encoders(output_path=encoders_pickle_path,
                              cat_features_encoders=cat_features_encoders,
                              labels_class_weights=labels_class_weights)
    save_embedder(emb, embedder_out_fp)


def script_prepare_data(content_folder: str = 'home/lemeiz/content_refine'):
    # loaded_data = load_contents_from_folder(content_folder, subset=False)
    # preprocessed_data = preprocess_article_data(loaded_data)
    # data, cat_features_encoders, labels_class_weights = process_naml_cat_features(preprocessed_data)
    data = pd.read_csv('data/finished_article_data.csv', index_col=[0], na_filter=False)
    corpus = generate_corpus(data)
    tokenize_dataset(data)
    emb = build_embedder(corpus)
    data.title, data.body = vectorize_textual_data(data, emb)
    return data  # , cat_features_encoders, labels_class_weights, emb


def main():
    #execute_full_data_preperation(out_fp='data/finished_article_data.csv')
    script_prepare_data()

if __name__ == '__main__':
    main()

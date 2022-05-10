import os.path

import pandas as pd
from tqdm import tqdm
import os
from time import time
import globals

import tokenizer_embedder


def resave_data_to_df(fp_list: list = None, output_dir: str = 'interaction_data'):
    for fp in tqdm(fp_list):
        df = pd.read_json(fp, lines=True)
        # df.drop(['query', 'referrerSearchEngine', 'referrerSocialNetwork', 'referrerQuery', 'os', 'referrerQuery', 'deviceType', 'referrerHostClass'], axis=1)
        df.to_csv(os.path.join(output_dir, f'{fp.split("/")[-1]}.csv'))
    return True


def load_single_file_from_json(fp: str, subset: bool = False):
    if subset:
        n_rows = 5000
    else:
        n_rows = None

    return pd.read_json(fp, lines=True, nrows=n_rows)


def load_data_from_json(fp_list: list = None, subset: bool = True):
    if subset:
        n_rows = 5000
    else:
        n_rows = None

    df_list = []
    for fp in tqdm(fp_list):
        file_df = pd.read_json(fp, lines=True, nrows=n_rows)
        cols = ['eventId', 'city', 'url', 'country', 'region',
                'time', 'userId', 'activeTime', 'canonicalUrl', 'id', 'publishtime']
        file_df = file_df[cols]
        file_df.drop_duplicates('eventId', inplace=True)
        df_list.append(file_df)
        if subset:
            break

    x = pd.concat(df_list, ignore_index=True, axis=0)
    return x


def load_data_from_json_to_file(fp_list: list = None, subset: bool = True):
    if subset:
        n_rows = 5000
    else:
        n_rows = None

    df_list = []
    for fp in tqdm(fp_list):
        file_df = pd.read_json(fp, lines=True, nrows=n_rows)
        cols = ['eventId', 'city', 'url', 'country', 'region',
                'time', 'userId', 'activeTime', 'canonicalUrl', 'id', 'publishtime']
        file_df = file_df[cols]
        file_df.drop_duplicates('eventId', inplace=True)
        df_list.append(file_df)
        if subset:
            break

    x = pd.concat(df_list, ignore_index=True, axis=0)
    x.to_csv('interaction_data/test.csv')
    return x


def load_data_from_csv(fp: str):
    return pd.read_csv(fp, index_col=0)


def load_article_data_manifest(article_df: pd.DataFrame) -> dict:
    return dict(article_df[['url', 'id_encoded']].apply(lambda x: (x['url'], x['id_encoded']), axis=1).values)


# Retrieves article id from its canonical URL (sometimes articleId don't match with articles, but canonical URL do)
def get_article_id_encoded_from_url(canonical_url, article_data_manifest: dict):
    if isinstance(canonical_url, list):
        x = []
        for url in canonical_url:
            x.append(int(article_data_manifest[url])) if url in article_data_manifest else x.append(None)
    elif isinstance(canonical_url, str):
        if canonical_url in article_data_manifest:
            return int(article_data_manifest[canonical_url])
        return None
    elif isinstance(canonical_url, pd.Series):
        return canonical_url.apply(lambda x: int(article_data_manifest[x]) if x in article_data_manifest else None)
    else:
        return None


def vectorize_interaction_locations(df: pd.DataFrame, key2index: dict) -> pd.Series:
    def vec_loc(row):
        x = key2index.get(row['city'])
        if x:
            return x
        x = key2index.get(row['region'])
        if x:
            return x
        x = key2index.get(globals.country_codes.get(row['country']))
        if x:
            return x
        return 0

    return df.apply(vec_loc, axis=1)


def preprocess_interactions(df: pd.DataFrame, manifest: dict, vectorizer: dict):
    df['article_id'] = get_article_id_encoded_from_url(canonical_url=df.canonicalUrl, article_data_manifest=manifest)
    df = df.dropna(subset=['article_id'])
    df.article_id = df.article_id.apply(int)
    df['location'] = vectorize_interaction_locations(df[['city', 'region', 'country']], vectorizer)
    df = df[['eventId', 'location', 'time', 'userId', 'article_id']]
    return df


def full_send(interaction_files: [str], article_fp: str = 'data/clean_encoded_data.csv',
              out_dir: str = 'interaction_data/preproc', embedder_path: str = 'data/embedder/embedder_model',
              test=False):
    t = time()
    articles = pd.read_csv(article_fp)
    manifest = load_article_data_manifest(article_df=articles)
    embedder = tokenizer_embedder.load_embedder(embedder_path)
    for file in tqdm(interaction_files):
        t1 = time()
        df = load_single_file_from_json(file, subset=test)
        df = preprocess_interactions(df, manifest=manifest,
                                     vectorizer=embedder.wv.key_to_index)  # does NOT remove those without sufficient articles
        df.to_csv(os.path.join(out_dir, f'{file.split("/")[-1]}.csv'))
        print(f'File time ({file.split("/")[-1]}): {time() - t1}')
        if test:
            break
    print(f'Total time: {time() - t}')


def idun():
    manifest = load_article_data_manifest(article_df=pd.read_csv('clean_encoded_data.csv'))
    for file in os.listdir('interaction_data'):
        df = load_single_file_from_json(file, subset=False)
        df['article_id'] = get_article_id_encoded_from_url(canonical_url=df.canonicalUrl,
                                                           article_data_manifest=manifest)
        df = df.dropna(subset=['article_id'])
        df.article_id = df.article_id.apply(int)


def main():
    full_send(interaction_files=['one_week/20170101'], test=False)


if __name__ == '__main__':
    main()

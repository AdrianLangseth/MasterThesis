import os.path
from os import mkdir
import warnings
import pandas as pd
import content_preprocessing
import globals
import interaction_preprocessing
import tokenizer_embedder
from tqdm import tqdm

warnings.filterwarnings("ignore")


def main(content_dir_fp: str = 'content_data',
         content_out_fp:str = 'content_data.csv',
         interaction_dir_fp: str = 'interaction_data',
         inter_out_dir: str = 'preprocessed_interaction_data',
         embedder_out_fp: str = 'data/embedder/embedder_model',
         ):
    """
    :param embedder_out_fp:
    :param content_dir_fp:
    :param content_out_fp:
    :param interaction_dir_fp:
    :param inter_out_dir:
    :return:
    """

    if not os.path.isdir(interaction_dir_fp):
        mkdir(inter_out_dir)
    # Load content data
    loaded_content_data = content_preprocessing.load_contents_from_folder(content_dir_fp, subset=False)
    # preprocess content data
    preprocessed_data = content_preprocessing.preprocess_article_data(loaded_content_data)
    content_data, cat_features_encoders, labels_class_weights = content_preprocessing.process_naml_cat_features(
        preprocessed_data)
    # Build corpus
    corpus = tokenizer_embedder.generate_corpus(content_data)
    # Tokenize data
    tokenizer_embedder.tokenize_dataset(content_data)
    # Build embedder
    gensim_embedder = tokenizer_embedder.build_embedder(corpus)
    # Vectorize textuals
    content_data.title, content_data.body = tokenizer_embedder.vectorize_textual_data(content_data, gensim_embedder)
    # save content_data
    content_data.to_csv(content_out_fp)
    # generate manifest
    manifest = interaction_preprocessing.load_article_data_manifest(article_df=content_data)
    # iterate over all files in interaction_path
    start_datapoints = 0
    end_datapoints = 0
    for file in tqdm(os.listdir(interaction_dir_fp)):
        df: pd.DataFrame = interaction_preprocessing.load_single_file_from_json(file, subset=False)
        start_datapoints += df.size
        df = interaction_preprocessing.preprocess_interactions(df, manifest=manifest,
                                                               vectorizer=gensim_embedder.wv.key_to_index)  # does NOT remove those without sufficient articles
        end_datapoints += df.size
        df.to_csv(os.path.join(inter_out_dir, f'{file.split("/")[-1]}.csv'))
    tokenizer_embedder.save_embedder(gensim_embedder, embedder_out_fp)
    print(f'Preprocessing complete.')
    print(f'Started with {start_datapoints} datapoints.')
    print(f'Ended with {end_datapoints} datapoints.')
    print(f'Saved content data to file: {content_out_fp}')
    print(f'Saved interaction data to folder: {inter_out_dir}')
    print(f'Saved embedder to folder: {embedder_out_fp}')



def combine_df_and_remove_infrequent_users(folder_fp, out_fp):
    df = pd.DataFrame()
    # combine
    for csv in tqdm(os.listdir(folder_fp)):
        try:
            df = pd.concat([df, interaction_preprocessing.load_data_from_csv(os.path.join(folder_fp, csv))])
        except (UnicodeDecodeError, pd.errors.ParserError) as e:
            print(e)
            print(f'Tried to load_csv from {csv}. Failed.')
            raise e

    filtered_df = df.groupby('userId').filter(lambda x: len(x) >= globals.data_params['min_interactions'])
    filtered_df.to_csv(out_fp)
    print(f'Combined all dataframes into one and filtered out all with interactions_num < {globals.data_params["min_history_size"]}.')
    print(f'Started with {df.shape[0]} interactions, ended with {filtered_df.shape[0]} interactions.')
    print(f'Final shape of dataframe: {filtered_df.shape}')
    return df






if __name__ == '__main__':
    main()
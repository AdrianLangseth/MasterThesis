import full_preprocessing
import os.path
import os
import random
import time
from ast import literal_eval

import numpy as np
import pandas as pd
import sklearn.model_selection
import wandb
from tqdm import tqdm
from wandb.keras import WandbCallback
import tensorflow as tf

import globals
import interaction_preprocessing
import model


class AdressaIterator:
    def __init__(self,
                 article_fp='data/clean_encoded_data.csv',
                 interaction_fp='interaction_data/preproc/20170106.csv'):
        """
        :param batch_size:
        :param article_fp: file path to article information CSV
        :param interaction_fp: filepath to CSV data describing the user interactions. Should be on form:
        +------------------+------------------------+--------------------------+
        |Current article_id| historical article_ids | User Interaction Location|
        +------------------+------------------------+--------------------------+
        """
        self.data_shape = None
        self.batch_size = globals.learning_params['batch_size']

        self.interaction_fp = interaction_fp

        self.__init_article_data(article_fp)
        self.__init_interaction_data(interaction_fp)

        self.main_generation_method = self.ret_from_interaction_log_with_loc_pos

        self.article_popularity = self.interaction_data.article_id.value_counts(normalize=True)

    def __init_article_data(self, article_fp):
        self.article_data = interaction_preprocessing.load_data_from_csv(article_fp)
        self.article_manifest = None  # interaction_preprocessing.load_article_data_manifest(self.article_data)
        # ToDo: Remove article manifest usage.

    def __init_interaction_data(self, interaction_fp):
        self.interaction_data = interaction_preprocessing.load_data_from_csv(interaction_fp)

    def ret_from_interaction_log_without_loc(self) -> ((np.ndarray, np.ndarray), np.ndarray):
        """
        returns data from interaction with user history. Should also give a negative sample.
        Interaction data is on form (event_id:int, location:float, time:int, user:str, article_id:int
        :return: interaction on form (candidate_data, user_history_data) and y value indicating whether it is read.
        :rtype: tuple
        :shape candidate_data: (batch_size, title_size + body_size + 2)
        :shape user_history_data: (batch_size, history_size, title_size + body_size + 2)
        :shape y: (batch_size)
        """
        concat_candidate_data = None
        concat_user_history_data = None
        concat_y = None
        df = self.interaction_data

        total_data_count = 0

        for user in tqdm(df['userId'].unique()):
            user_interactions = df[df['userId'] == user].sort_values(by='time', axis=0, ascending=True)
            if user_interactions.shape[0] - globals.data_params['min_history_size'] < 1:
                continue

            candidate_data = np.zeros(((user_interactions.shape[0] - globals.data_params['min_history_size']) * 2,
                                       globals.data_params['article_size']))
            user_history_data = np.zeros(((user_interactions.shape[0] - globals.data_params['min_history_size']) * 2,
                                          globals.data_params['max_no_documents_in_user_profile'],
                                          globals.data_params['article_size']))
            y = np.zeros((user_interactions.shape[0] - globals.data_params['min_history_size']) * 2)

            data_count = 0
            history = []
            for idx, interaction in user_interactions.iterrows():
                if len(history) < globals.data_params['min_history_size']:
                    history.append(interaction)
                    continue

                temp_history_article_data = self.__get_article_data(history)

                # Add data of positive sample to batches
                candidate_data[data_count, :] = self.__get_article_data(
                    interaction)  # Any error on slicing here may stem from that article data is wrong in that they should be uniform in length and always abide by globals.article_size, and should already be formed by this in the data file.
                user_history_data[data_count, :len(temp_history_article_data),
                :] = temp_history_article_data  # add data available and leave rest zeroes.
                y[data_count] = 1
                data_count += 1  # Increment pointer
                total_data_count += 1

                # add negative sample with same history
                candidate_data[data_count, :] = self.__get_negative_sample(interaction)
                user_history_data[data_count, :len(temp_history_article_data), :] = temp_history_article_data
                y[data_count] = 0
                data_count += 1

                if len(history) > globals.data_params['max_no_documents_in_user_profile']:
                    raise ValueError(
                        f'History size should not exceed max size. history size was {len(history)}, max is {globals.data_params["max_no_documents_in_user_profile"]}. This is likely an issue with batch_size and adding the negative sample in addition to the positive sample without control ovr the batch size. Right now assumes that batch_size is even. If this is not the case, this will not go well.')

                if len(history) == globals.data_params['max_no_documents_in_user_profile']:
                    history.pop(0)
                history.append(interaction)

            if concat_user_history_data is None:
                concat_user_history_data = user_history_data
                concat_candidate_data = candidate_data
                concat_y = y
            else:
                concat_user_history_data = np.concatenate([concat_user_history_data, user_history_data], axis=0)
                concat_candidate_data = np.concatenate([concat_candidate_data, candidate_data], axis=0)
                concat_y = np.concatenate([concat_y, y], axis=0)

        return (concat_candidate_data, concat_user_history_data), concat_y

    def ret_from_interaction_log_with_loc(self) -> ((np.ndarray, np.ndarray), np.ndarray):
        """
        returns data from interaction with user history. Should also give a negative sample.
        Interaction data is on form (event_id:int, location:float, time:int, user:str, article_id:int
        :return: interaction on form (candidate_data, user_history_data) and y value indicating whether it is read.
        :rtype: tuple
        :shape candidate_data: (batch_size, title_size + body_size + 2)
        :shape user_history_data: (batch_size, history_size, title_size + body_size + 2)
        :shape y: (batch_size)
        """
        concat_candidate_data = None
        concat_user_history_data = None
        concat_y = None
        # pull interaction_data and filter out those with less than usable interactions
        df = self.interaction_data.groupby('userId').filter(lambda x: len(x) > globals.data_params['min_interactions'])

        total_data_count = 0

        for user in tqdm(df['userId'].unique()):
            user_interactions = df[df['userId'] == user].sort_values(by='time', axis=0, ascending=True)
            # if user_interactions.shape[0] - globals.data_params['min_history_size'] < 1:
            #    continue

            candidate_data = np.zeros(((user_interactions.shape[0] - globals.data_params['min_history_size']) * 2,
                                       globals.data_params['article_size'] + globals.data_params[
                                           'max_locations_per_article']))
            user_history_data = np.zeros(((user_interactions.shape[0] - globals.data_params['min_history_size']) * 2,
                                          globals.data_params['max_no_documents_in_user_profile'],
                                          globals.data_params['article_size'] + globals.data_params[
                                              'max_locations_per_article']))
            y = np.zeros((user_interactions.shape[0] - globals.data_params['min_history_size']) * 2)

            data_count = 0
            history = []
            for idx, interaction in user_interactions.iterrows():
                if len(history) < globals.data_params['min_history_size']:
                    history.append(interaction)
                    continue

                temp_history_article_data = self.__get_article_data(history, loc=True)

                # Add data of positive sample to batches
                candidate_data[data_count, :] = self.__get_article_data(interaction, loc=True)
                user_history_data[data_count, :len(temp_history_article_data),
                :] = temp_history_article_data  # add data available and leave rest zeroes.
                y[data_count] = 1
                data_count += 1  # Increment pointer
                total_data_count += 1

                # add negative sample with same history
                candidate_data[data_count, :] = self.__get_negative_sample(interaction, loc=True)
                user_history_data[data_count, :len(temp_history_article_data), :] = temp_history_article_data
                y[data_count] = 0
                data_count += 1

                if len(history) > globals.data_params['max_no_documents_in_user_profile']:
                    raise ValueError(
                        f'History size should not exceed max size. history size was {len(history)}, max is {globals.data_params["max_no_documents_in_user_profile"]}. This is likely an issue with batch_size and adding the negative sample in addition to the positive sample without control ovr the batch size. Right now assumes that batch_size is even. If this is not the case, this will not go well.')

                if len(history) == globals.data_params['max_no_documents_in_user_profile']:
                    history.pop(0)
                history.append(interaction)

            if concat_user_history_data is None:
                concat_user_history_data = user_history_data
                concat_candidate_data = candidate_data
                concat_y = y
            else:
                concat_user_history_data = np.concatenate([concat_user_history_data, user_history_data], axis=0)
                concat_candidate_data = np.concatenate([concat_candidate_data, candidate_data], axis=0)
                concat_y = np.concatenate([concat_y, y], axis=0)

        return (concat_candidate_data, concat_user_history_data), concat_y

    def ret_from_interaction_log_with_loc_pos(self) -> ((np.ndarray, np.ndarray), np.ndarray):
        """
        returns data from interaction with user history. Should also give a negative sample.
        Interaction data is on form (event_id:int, location:float, time:int, user:str, article_id:int
        :return: interaction on form (candidate_data, user_history_data) and y value indicating whether it is read.
        :rtype: tuple
        :shape candidate_data: (batch_size, title_size + body_size + 2)
        :shape user_history_data: (batch_size, history_size, title_size + body_size + 2)
        :shape y: (batch_size)
        """
        concat_candidate_data = None
        concat_user_history_data = None
        concat_y = None
        # pull interaction_data and filter out those with less than usable interactions
        df = self.interaction_data.groupby('userId').filter(lambda x: len(x) > globals.data_params['min_interactions'])

        total_data_count = 0

        for user in tqdm(df['userId'].unique()):
            user_interactions = df[df['userId'] == user].sort_values(by='time', axis=0, ascending=True)
            # if user_interactions.shape[0] - globals.data_params['min_history_size'] < 1:
            #    continue

            candidate_data = np.zeros(((user_interactions.shape[0] - globals.data_params['min_history_size']) * 2,
                                       globals.data_params['article_size'] + globals.data_params[
                                           'max_locations_per_article'] + 1))
            user_history_data = np.zeros(((user_interactions.shape[0] - globals.data_params['min_history_size']) * 2,
                                          globals.data_params['max_no_documents_in_user_profile'],
                                          globals.data_params['article_size'] + globals.data_params[
                                              'max_locations_per_article'] + 1))
            y = np.zeros((user_interactions.shape[0] - globals.data_params['min_history_size']) * 2)

            data_count = 0
            history = []
            history_ids = []
            for idx, interaction in user_interactions.iterrows():
                if len(history) < globals.data_params['min_history_size']:
                    history.append(interaction)
                    history_ids.append(interaction.article_id)
                    continue

                temp_history_article_data = self.__get_article_data(history, loc=True, pos=True)

                # Add data of positive sample to batches
                candidate_data[data_count, :] = self.__get_article_data(interaction, loc=True, pos=True)
                user_history_data[data_count, :len(temp_history_article_data),
                :] = temp_history_article_data  # add data available and leave rest zeroes.
                y[data_count] = 1
                data_count += 1  # Increment pointer
                total_data_count += 1

                # add negative sample with same history
                candidate_data[data_count, :] = self.__get_negative_sample(interaction,
                                                                           loc=True,
                                                                           pos=True,
                                                                           history=history_ids
                                                                           )
                user_history_data[data_count, :len(temp_history_article_data), :] = temp_history_article_data
                y[data_count] = 0
                data_count += 1

                if len(history) > globals.data_params['max_no_documents_in_user_profile']:
                    raise ValueError(
                        f'History size should not exceed max size. history size was {len(history)}, max is {globals.data_params["max_no_documents_in_user_profile"]}. This is likely an issue with batch_size and adding the negative sample in addition to the positive sample without control ovr the batch size. Right now assumes that batch_size is even. If this is not the case, this will not go well.')

                if len(history) == globals.data_params['max_no_documents_in_user_profile']:
                    history.pop(0)
                    history_ids.pop(0)
                history.append(interaction)
                history_ids.append(interaction.article_id)

            if concat_user_history_data is None:
                concat_user_history_data = user_history_data
                concat_candidate_data = candidate_data
                concat_y = y
            else:
                concat_user_history_data = np.concatenate([concat_user_history_data, user_history_data], axis=0)
                concat_candidate_data = np.concatenate([concat_candidate_data, candidate_data], axis=0)
                concat_y = np.concatenate([concat_y, y], axis=0)

        return (concat_candidate_data, concat_user_history_data), concat_y

    def ret_from_interaction_log_with_loc_pos_opt(self) -> ((np.ndarray, np.ndarray), np.ndarray):
        """
        returns data from interaction with user history. Should also give a negative sample.
        Interaction data is on form (event_id:int, location:float, time:int, user:str, article_id:int
        :return: interaction on form (candidate_data, user_history_data) and y value indicating whether it is read.
        :rtype: tuple
        :shape candidate_data: (batch_size, title_size + body_size + 2)
        :shape user_history_data: (batch_size, history_size, title_size + body_size + 2)
        :shape y: (batch_size)
        """
        concat_candidate_data = None
        concat_user_history_data = None
        concat_y = None
        # pull interaction_data and filter out those with less than usable interactions
        df = self.interaction_data.groupby('userId').filter(lambda x: len(x) > globals.data_params['min_interactions'])

        total_data_count = 0

        for user in tqdm(df['userId'].unique()):
            user_interactions = df[df['userId'] == user].sort_values(by='time', axis=0, ascending=True)
            # if user_interactions.shape[0] - globals.data_params['min_history_size'] < 1:
            #    continue

            candidate_data = np.zeros(((user_interactions.shape[0] - globals.data_params['min_history_size']) * 2,
                                       globals.data_params['article_size'] + globals.data_params[
                                           'max_locations_per_article'] + 1))
            user_history_data = np.zeros(((user_interactions.shape[0] - globals.data_params['min_history_size']) * 2,
                                          globals.data_params['max_no_documents_in_user_profile'],
                                          globals.data_params['article_size'] + globals.data_params[
                                              'max_locations_per_article'] + 1))
            y = np.zeros((user_interactions.shape[0] - globals.data_params['min_history_size']) * 2)

            data_count = 0
            history = []  # optimized to be a list of iteraction_article_data
            start_hist = []
            for idx, interaction in user_interactions.iterrows():
                if len(start_hist) < globals.data_params['min_history_size']:
                    start_hist.append(interaction)
                    continue
                elif len(start_hist) == globals.data_params['min_history_size']:
                    history = self.__get_article_data(start_hist, loc=True, pos=True)

                cand_data = self.__get_article_data(interaction, loc=True, pos=True)
                temp_history_article_data = np.vstack(history)

                # Add data of positive sample to batches
                candidate_data[data_count, :] = cand_data
                user_history_data[data_count, :len(temp_history_article_data),
                :] = temp_history_article_data  # add data available and leave rest zeroes.
                y[data_count] = 1
                data_count += 1  # Increment pointer
                total_data_count += 1

                # add negative sample with same history
                candidate_data[data_count, :] = self.__get_negative_sample(interaction, loc=True, pos=True)
                user_history_data[data_count, :len(temp_history_article_data), :] = temp_history_article_data
                y[data_count] = 0
                data_count += 1

                if len(history) > globals.data_params['max_no_documents_in_user_profile']:
                    raise ValueError(
                        f'History size should not exceed max size. history size was {len(history)}, max is {globals.data_params["max_no_documents_in_user_profile"]}. This is likely an issue with batch_size and adding the negative sample in addition to the positive sample without control ovr the batch size. Right now assumes that batch_size is even. If this is not the case, this will not go well.')

                if len(history) == globals.data_params['max_no_documents_in_user_profile']:
                    history.pop(0)
                history.append(cand_data)

            if concat_user_history_data is None:
                concat_user_history_data = user_history_data
                concat_candidate_data = candidate_data
                concat_y = y
            else:
                concat_user_history_data = np.concatenate([concat_user_history_data, user_history_data], axis=0)
                concat_candidate_data = np.concatenate([concat_candidate_data, candidate_data], axis=0)
                concat_y = np.concatenate([concat_y, y], axis=0)

        return (concat_candidate_data, concat_user_history_data), concat_y

    def __get_article_data(self, interaction, loc: bool = False, pos: bool = False):
        """
        Does the information pull from interaction to usable data.
        If interaction is a single interaction (one series) it is given as a single numpy_1d array.
        - This is safe, cannot fuck this up.
        If interaction is a list of interactions the data is given as a 2d numpy array.
        - This must be done by fist making np.zeroes and filling data on top.
        :param interaction: Dataframe object or Series
        :return: data on extended form (title body cat subcat) with shape (N, article_size) where N is the amount of
        interactions.
        """
        if isinstance(interaction, pd.Series):  # Single interaction
            art_ids = [interaction['article_id']]
        else:  # multiple interactions
            art_ids = [inter['article_id'] for inter in interaction]

        texts = self.article_data.loc[art_ids][['title', 'body']]
        cats = self.article_data.loc[art_ids][['category0_encoded', 'category1_encoded']]
        titles = texts['title'].apply(literal_eval)  # .apply(lambda x: literal_eval(x) if type(x) == str else x)
        bodies = texts['body'].apply(literal_eval)  # .apply(lambda x: literal_eval(x) if type(x) == str else x)
        title_padded = titles.apply(lambda x: x[:globals.data_params['title_size']]).apply(
            lambda c: np.pad(c, pad_width=(0, globals.data_params['title_size'] - len(c))))
        body_padded = bodies.apply(lambda x: x[:globals.data_params['body_size']]).apply(
            lambda c: np.pad(c, pad_width=(0, globals.data_params['body_size'] - len(c))))

        all = [np.vstack(title_padded.to_numpy()), np.vstack(body_padded.to_numpy()), np.atleast_2d(cats.to_numpy())]

        if loc:
            locs = self.article_data.loc[art_ids]['locations_encoded'].apply(literal_eval)
            locs_padded = locs.apply(lambda x: x[:globals.data_params['max_locations_per_article']]).apply(
                lambda c: np.pad(c, pad_width=(0, globals.data_params['max_locations_per_article'] - len(c))))
            all.append(np.vstack(locs_padded.to_numpy()))

        if pos:
            if type(interaction) != list:  # This can be combined with the check earlier
                interaction = [interaction]
            pos = pd.concat(interaction, axis=1).loc['location']
            pos.apply(lambda x: 0 if x is None else x)
            all.append(np.expand_dims(pos.to_numpy(), axis=-1))

        return np.hstack(all)

    def __get_content_data_from_id(self, art_ids: list):
        texts = self.article_data.loc[art_ids][['title', 'body']]
        cats = self.article_data.loc[art_ids][['category0_encoded', 'category1_encoded']]
        titles = texts['title'].apply(literal_eval)
        bodies = texts['body'].apply(literal_eval)
        title_padded = titles.apply(lambda x: x[:globals.data_params['title_size']]).apply(
            lambda c: np.pad(c, pad_width=(0, globals.data_params['title_size'] - len(c))))
        body_padded = bodies.apply(lambda x: x[:globals.data_params['body_size']]).apply(
            lambda c: np.pad(c, pad_width=(0, globals.data_params['body_size'] - len(c))))

        return np.hstack(
            [np.vstack(title_padded.to_numpy()), np.vstack(body_padded.to_numpy()), np.atleast_2d(cats.to_numpy)])

    def __get_negative_sample(self, interaction, loc: bool = False, pos: bool = False, pop=True, history: list = None):
        temp = interaction.copy(deep=True)
        if not pop:
            random_article_index = random.choice(self.article_data.index.to_list())
            while random_article_index == temp.article_id:
                random_article_index = random.choice(self.article_data.index.to_list())
            temp.article_id = random_article_index
            return self.__get_article_data(interaction=temp, loc=loc, pos=pos)

        random_article_index = interaction.article_id
        while random_article_index == interaction.article_id or random_article_index in history:
            random_article_index = self.article_popularity.sample(1, weights=self.article_popularity.values,
                                                                  random_state=globals.learning_params[
                                                                      'seed']).index[0]
        temp.article_id = random_article_index
        return self.__get_article_data(interaction=temp, loc=loc, pos=pos)

    def load_data_to_npy(self, int_out_dir):
        (x_c, x_u), y = self.main_generation_method()

        if not os.path.isdir(f'training_data/{int_out_dir}'):
            os.makedirs(f'training_data/{int_out_dir}')
        np.save(f'training_data/{int_out_dir}/cand.npy', x_c)
        np.save(f'training_data/{int_out_dir}/hist.npy', x_u)
        np.save(f'training_data/{int_out_dir}/y.npy', y)


def main_gen():
    genny = AdressaIterator(interaction_fp='interaction_data/preproc/20170106.csv')
    (x_c, x_u), y = genny.main_generation_method()

    # np.save('training_data/20170105/cand.npy', x_c)
    # np.save('training_data/20170105/hist.npy', x_u)
    # np.save('training_data/20170105/y.npy', y)

    cand_t, cand_v, hist_t, hist_v, y_t, y_v = sklearn.model_selection.train_test_split(x_c, x_u, y)
    m = model.naml_both('keras')
    m.fit(x=[cand_t, hist_t],
          y=y_t,
          batch_size=globals.learning_params['batch_size'],
          epochs=globals.learning_params['epochs'],
          validation_data=([cand_v, hist_v], y_v),
          callbacks=[WandbCallback()],
          )
    return m


def load_dataset_npy(folder='20170101'):
    if isinstance(folder, str):
        x_c = np.load(f'training_data/{folder}/cand.npy').astype(int)
        x_u = np.load(f'training_data/{folder}/hist.npy').astype(int)
        y = np.load(f'training_data/{folder}/y.npy').astype(int)
    elif isinstance(folder, list):
        x_c, x_u, y = None, None, None
        for date in folder:
            if x_c is None:
                x_c = np.load(f'training_data/{date}/cand.npy')
                x_u = np.load(f'training_data/{date}/hist.npy')
                y = np.load(f'training_data/{date}/y.npy')
            else:
                x_c = np.concatenate([x_c, np.load(f'training_data/{date}/cand.npy')])
                x_u = np.concatenate([x_u, np.load(f'training_data/{date}/hist.npy')])
                y = np.concatenate([y, np.load(f'training_data/{date}/y.npy')])
    else:
        raise ValueError(f'Must be either string or list. Was {type(folder)}')
    return (x_c, x_u), y


def load_full_dataset_from_interactions(train: list, validation: list, test: list):
    #
    for l in [train, validation, test]:
        for file in l:
            os.path.exists(file)

    full_preprocessing.combine_df_and_remove_infrequent_users()

    pass


def load_full_dataset_from_numpy_files(train, validation, test):
    for file in [train, validation, test]:
        os.path.exists(file)

    pass


def script(
        train_list,
        validation_list,
        test_list,
        in_folder='preprocessed_interaction_data_min25',
        out_folder='combined_interaction_data/test',
        article_fp='content_data.csv',
        save=True,
        save_path='training_data/full_roll'
):
    # start: encoded interaction_logs

    assert os.path.exists(article_fp) and os.path.exists(in_folder)

    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    train_list = [os.path.join(in_folder, i) for i in train_list]
    validation_list = [os.path.join(in_folder, i) for i in validation_list]
    test_list = [os.path.join(in_folder, i) for i in test_list]

    # Combine
    train: pd.DataFrame = full_preprocessing.combine_df_and_remove_infrequent_users_from_list(train_list)
    validation: pd.DataFrame = full_preprocessing.combine_df_and_remove_infrequent_users_from_list(validation_list)
    test: pd.DataFrame = full_preprocessing.combine_df_and_remove_infrequent_users_from_list(test_list)

    # save
    train.to_csv(os.path.join(out_folder, 'train.csv'))
    validation.to_csv(os.path.join(out_folder, 'validation.csv'))
    test.to_csv(os.path.join(out_folder, 'test.csv'))

    # Data Gen
    g = AdressaIterator(article_fp=article_fp, interaction_fp=os.path.join(out_folder, 'train.csv'))
    (X_train, y_train) = g.ret_from_interaction_log_with_loc_pos()

    g = AdressaIterator(article_fp=article_fp, interaction_fp=os.path.join(out_folder, 'validation.csv'))
    (X_val, y_val) = g.ret_from_interaction_log_with_loc_pos()

    g = AdressaIterator(article_fp=article_fp, interaction_fp=os.path.join(out_folder, 'test.csv'))
    (X_test, y_test) = g.ret_from_interaction_log_with_loc_pos()

    if save:
        os.makedirs(f'{save_path}/train/', exist_ok=True)
        os.makedirs(f'{save_path}/val/', exist_ok=True)
        os.makedirs(f'{save_path}/test/', exist_ok=True)
        np.save(f'{save_path}/train/x_c', X_train[0])
        np.save(f'{save_path}/train/x_u', X_train[1])
        np.save(f'{save_path}/train/y', y_train)

        np.save(f'{save_path}/val/x_c', X_val[0])
        np.save(f'{save_path}/val/x_u', X_val[1])
        np.save(f'{save_path}/val/y', y_val)

        np.save(f'{save_path}/test/x_c', X_test[0])
        np.save(f'{save_path}/test/x_u', X_test[1])
        np.save(f'{save_path}/test/y', y_test)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    # save_to_numpy or return to model


if __name__ == '__main__':
    train = [f'201701{i:02d}' for i in range(1, 32)]
    train.extend([f'201702{i:02d}' for i in range(1, 29)])
    valid = [f'201703{i:02d}' for i in range(1, 15)]
    test = [f'201703{i:02d}' for i in range(15, 29)]

import os.path
import os
import random
import time
from ast import literal_eval

import numpy as np
import pandas as pd
import sklearn.model_selection
from tqdm import tqdm
from wandb.keras import WandbCallback

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

    def __init_article_data(self, article_fp):
        self.article_data = interaction_preprocessing.load_data_from_csv(article_fp)
        self.article_manifest = None  # interaction_preprocessing.load_article_data_manifest(self.article_data)
        # ToDo: Remove article manifest usage.

    def __init_interaction_data(self, interaction_fp):
        self.interaction_data = interaction_preprocessing.load_data_from_csv(interaction_fp)

    def generate_from_interaction_log_without_loc(self):
        """
        Yields a batch of data from interaction with user history. Should also give a negative sample.
        Interaction data is on form (event_id:int, location:float, time:int, user:str, article_id:int
        :return: interaction on form (candidate_data, user_history_data) and y value indicating whether it is read.
        :rtype: tuple
        :shape candidate_data: (batch_size, title_size + body_size + 2)
        :shape user_history_data: (batch_size, history_size, title_size + body_size + 2)
        :shape y: (batch_size)
        """
        candidate_data = np.zeros((globals.learning_params['batch_size'],
                                   globals.data_params['article_size']))
        user_history_data = np.zeros((globals.learning_params['batch_size'],
                                      globals.data_params['max_no_documents_in_user_profile'],
                                      globals.data_params['article_size']))
        y = np.zeros(globals.learning_params['batch_size'])

        df = self.interaction_data
        data_count = 0
        batches = 0
        t = time.time()
        while True:
            for user in df['userId'].unique():
                user_interactions = df[df['userId'] == user].sort_values(by='time', axis=0, ascending=True)
                if user_interactions.shape[0] < globals.data_params['min_history_size'] + 1:
                    continue
                history = []
                for idx, interaction in user_interactions.iterrows():
                    if len(history) < globals.data_params['min_history_size']:
                        history.append(interaction)
                        continue

                    temp_history_article_data = self.__get_article_data(history)
                    # temp_history_array = np.zeros((globals.data_params['max_no_documents_in_user_profile'], globals.data_params['article_size']))
                    # temp_history_array[:temp_history_article_data.shape[0], :] = temp_history_article_data

                    # Add data of positive sample to batches
                    candidate_data[data_count, :] = self.__get_article_data(
                        interaction)  # Any error on slicing here may stem from that article data is wrong in that
                    # they should be uniform in length and always abide by globals.article_size, and should already
                    # be formed by this in the data file.
                    user_history_data[data_count, :len(temp_history_article_data),
                    :] = temp_history_article_data  # add data available and leave rest zeroes.
                    y[data_count] = 1
                    data_count += 1  # Increment pointer

                    # add negative sample with same history
                    candidate_data[data_count, :] = self.__get_random_negative_sample(interaction)
                    user_history_data[data_count, :len(temp_history_article_data), :] = temp_history_article_data
                    y[data_count] = 0  # Can drop, already 0 if initialized properly
                    data_count += 1

                    if len(history) > globals.data_params['max_no_documents_in_user_profile']:
                        raise ValueError(
                            f'History size should not exceed max size. history size was {len(history)}, max is {globals.data_params["max_no_documents_in_user_profile"]}. This is likely an issue with batch_size and adding the negative sample in addition to the positive sample without control ovr the batch size. Right now assumes that batch_size is even. If this is not the case, this will not go well.')

                    if len(history) == globals.data_params['max_no_documents_in_user_profile']:
                        history.pop(0)
                    history.append(interaction)

                    if data_count == len(y):
                        yield (candidate_data, user_history_data), y
                        # print(f'Batch of size {y.shape[0]} created. Took {time.time()-t} seconds.')
                        t = time.time()
                        # Reset
                        data_count = 0
                        candidate_data[:, :] = 0
                        user_history_data[:, :, :] = 0
                        y[:] = 0
                        batches += 1

                # if globals.data_params['min_history_size'] + idx < user_interactions.shape[0]: # My math might be off here. todo: check math
        # print(f'Done. Finished with {batches} batches over {epoch} epochs')

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
                candidate_data[data_count, :] = self.__get_random_negative_sample(interaction)
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
                candidate_data[data_count, :] = self.__get_random_negative_sample(interaction, loc=True)
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
            for idx, interaction in user_interactions.iterrows():
                if len(history) < globals.data_params['min_history_size']:
                    history.append(interaction)
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
                candidate_data[data_count, :] = self.__get_random_negative_sample(interaction, loc=True, pos=True)
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

    def __get_random_negative_sample(self, interaction, loc: bool = False, pos: bool = False):
        temp = interaction.copy(deep=True)
        # random_article = self.article_data.sample(n=1)
        random_aticle_index = random.choice(self.article_data.index.to_list())
        while random_aticle_index == temp.article_id:
            random_aticle_index = random.choice(self.article_data.index.to_list())
        temp.article_id = random_aticle_index
        return self.__get_article_data(interaction=temp, loc=loc, pos=pos)

    def load_data_to_npy(self):
        (x_c, x_u), y = self.ret_from_interaction_log_with_loc_pos()
        name = self.interaction_fp.split("/")[-1].split('.')[0]

        if not os.path.isdir(f'training_data/locpos/{name}'):
            os.makedirs(f'training_data/locpos/{name}')
        np.save(f'training_data/locpos/{name}/cand.npy', x_c)
        np.save(f'training_data/locpos/{name}/hist.npy', x_u)
        np.save(f'training_data/locpos/{name}/y.npy', y)


def main_gen():
    genny = AdressaIterator()
    (x_c, x_u), y = genny.ret_from_interaction_log_without_loc()

    np.save('training_data/20170106/cand.npy', x_c)
    np.save('training_data/20170106/hist.npy', x_u)
    np.save('training_data/20170106/y.npy', y)

    cand_t, cand_v, hist_t, hist_v, y_t, y_v = sklearn.model_selection.train_test_split(x_c, x_u, y)
    m = model.naml()
    m.fit(x=[cand_t, hist_t],
          y=y_t,
          batch_size=globals.learning_params['batch_size'],
          epochs=globals.learning_params['epochs'],
          validation_data=([cand_v, hist_v], y_v),
          # callbacks=[WandbCallback()],
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


def main_load(send_stats: bool = True):
    (x_c, x_u), y = load_dataset_npy('training_data')

    cand_t, cand_v, hist_t, hist_v, y_t, y_v = sklearn.model_selection.train_test_split(x_c, x_u, y, shuffle=False)
    m = model.naml('keras')
    callbacks = []
    if send_stats:
        callbacks.append(WandbCallback())

    m.fit(x=[cand_t, hist_t],
          y=y_t,
          batch_size=globals.learning_params['batch_size'],
          epochs=globals.learning_params['epochs'],
          validation_data=([cand_v, hist_v], y_v),
          callbacks=callbacks,
          )
    return m


if __name__ == '__main__':
    genny = AdressaIterator(interaction_fp='interaction_data/preproc/20170101.csv')
    genny.load_data_to_npy()
    print('Done')

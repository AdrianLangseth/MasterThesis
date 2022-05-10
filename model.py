import warnings

import sklearn
import wandb

import data_generator
from user_encoder import define_user_encoder_model
from news_encoder import build_news_encoder
from location_encoder import get_location_encoders, get_position_encoders, get_old_location_encoders
from tokenizer_embedder import get_embedding_layer_from_gensim
import tensorflow as tf
import globals
from wandb.keras import WandbCallback
from flexible_layers import Slice
import tensorflow_ranking as tfr

warnings.filterwarnings('ignore')


def naml(embedder: str = 'gensim'):
    # Build embedder, build news encoder based on embedder, build news encoder based on news encoder.
    if embedder == 'gensim':
        embedding_layer = get_embedding_layer_from_gensim()
    elif embedder == 'keras':
        embedding_layer = tf.keras.layers.Embedding(
            input_dim=globals.model_params['vocabulary_size'],
            output_dim=globals.model_params['word_embedding_dim'],
            embeddings_initializer="uniform",
            trainable=True,
        )
    else:
        raise ValueError(f'embedder parameter must be either "gensim" or "keras", but was {embedder}')
    news_encoder = build_news_encoder(embedding_layer=embedding_layer)
    user_encoder = define_user_encoder_model(
        input_shape=(globals.model_params['max_no_documents_in_user_profile'], globals.data_params['article_size']),
        encoding_operation=news_encoder
    )

    # Define inputs
    candidate = tf.keras.layers.Input(
        shape=(globals.model_params['title_size'] + globals.model_params['body_size'] + 2),
        dtype="int32")  # 2d
    history = tf.keras.layers.Input(shape=(globals.model_params['max_no_documents_in_user_profile'],
                                           globals.model_params['title_size'] + globals.model_params['body_size'] + 2),
                                    dtype="int32")  # 2d
    # Run candidate through news encoder and hostory through user encoder
    encoded_candidate = news_encoder(candidate)
    encoded_user = user_encoder(history)

    # Predict click probability
    full_model_output = tf.keras.layers.Dot(axes=-1)([encoded_candidate, encoded_user])

    # Define model & Compile
    model = tf.keras.models.Model([candidate, history], full_model_output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(globals.learning_params['learning_rate']),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.MeanSquaredError()],
    )

    return model


def naml_loc(embedder: str = 'gensim', click_predictor='dot'):
    # Build embedder, build news encoder based on embedder, build news encoder based on news encoder.
    if embedder == 'gensim':
        embedding_layer = get_embedding_layer_from_gensim()
    elif embedder == 'keras':
        embedding_layer = tf.keras.layers.Embedding(
            input_dim=globals.model_params['vocabulary_size'],
            output_dim=globals.model_params['word_embedding_dim'],
            embeddings_initializer="uniform",
            trainable=True,
        )
    else:
        raise ValueError(f'embedder parameter must be either "gensim" or "keras", but was {embedder}')
    news_encoder = build_news_encoder(embedding_layer=embedding_layer)
    user_encoder = define_user_encoder_model(
        input_shape=(globals.model_params['max_no_documents_in_user_profile'], globals.data_params['article_size']),
        encoding_operation=news_encoder
    )
    candidate_location_encoder, history_location_encoder = get_location_encoders(embedding_layer)

    # Define inputs

    candidate_location_encoder, history_location_encoder = get_location_encoders(embedding_layer)
    full_candidate = tf.keras.layers.Input(
        shape=(globals.data_params['article_size'] + globals.model_params['max_locations_per_article']),
        dtype="int32",
        name='candidate_Input',
    )  # 2d
    full_history = tf.keras.layers.Input(shape=(globals.model_params['max_no_documents_in_user_profile'],
                                                globals.data_params['article_size'] +
                                                globals.model_params['max_locations_per_article']),
                                         name='history_input',
                                         dtype="int32")

    # noinspection PyCallingNonCallable
    news_slicer = Slice(
        begin=0,
        size=globals.model_params['article_size'],
        name="slice_input_article"
    )
    # noinspection PyCallingNonCallable
    loc_slicer = Slice(
        begin=globals.model_params['article_size'],
        size=globals.data_params['max_locations_per_article'],
        name='slice_input_article_location_data',
    )
    # noinspection PyCallingNonCallable
    candidate = news_slicer(full_candidate)
    # noinspection PyCallingNonCallable
    candidate_loc = loc_slicer(full_candidate)

    history = tf.keras.layers.TimeDistributed(news_slicer)(full_history)
    history_loc = tf.keras.layers.TimeDistributed(loc_slicer)(full_history)

    # Slice candidate and history and pop them into their encoders.
    # Run candidate through news encoder and hostory through user encoder
    encoded_candidate = news_encoder(candidate)
    encoded_user = user_encoder(history)

    # noinspection PyUnboundLocalVariable
    encoded_cand_locs = candidate_location_encoder(candidate_loc)
    # noinspection PyUnboundLocalVariable
    encoded_hist_locs = history_location_encoder(history_loc)

    # Predict click probability
    news_score = tf.keras.layers.Dot(axes=-1)([encoded_candidate, encoded_user])
    loc_score = tf.keras.layers.Dot(axes=-1)([encoded_cand_locs, encoded_hist_locs])

    # full_model_output = tf.keras.layers.Dot(axes=-1)([news_score, loc_score])

    if click_predictor == 'neural':
        neural_inputs = tf.concat([news_score, loc_score], axis=-1)
        x = tf.keras.layers.Dense(4, activation='relu')(neural_inputs)
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        full_model_output = x
    elif click_predictor == 'dot':
        full_model_output = tf.keras.layers.Dot(axes=-1)([news_score, loc_score])
    else:
        raise NotImplementedError(f'Click predictor is implemented as Dot product ("dot") and neural network('
                                  f'"neural"). Could not find implementation for {click_predictor}.')

    # Define model & Compile
    model = tf.keras.models.Model([full_candidate, full_history], full_model_output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(globals.learning_params['learning_rate']),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.MeanSquaredError()],
    )

    return model


def naml_pos(embedder: str = 'gensim', click_predictor='neural', historical: bool = False, task: str = 'news'):
    # Check if the given task is valid
    if task not in get_tasks():
        raise ValueError(f'Given task must be one of the predefined but was {task}. Check docs for get_tasks().')

    # Build embedder, build news encoder based on embedder, build news encoder based on news encoder.
    if embedder == 'gensim':
        embedding_layer = get_embedding_layer_from_gensim()
    elif embedder == 'keras':
        embedding_layer = tf.keras.layers.Embedding(
            input_dim=globals.model_params['vocabulary_size'],
            output_dim=globals.model_params['word_embedding_dim'],
            embeddings_initializer="uniform",
            trainable=True,
        )
    else:
        raise ValueError(f'embedder parameter must be either "gensim" or "keras", but was {embedder}')
    news_encoder = build_news_encoder(embedding_layer=embedding_layer)
    user_encoder = define_user_encoder_model(
        input_shape=(globals.model_params['max_no_documents_in_user_profile'], globals.data_params['article_size']),
        encoding_operation=news_encoder
    )
    candidate_location_encoder, history_location_encoder = get_location_encoders(embedding_layer)

    user_pos_encoder, historical_pos_encoder, user_hist_combined_pos_encoder = get_position_encoders(embedding_layer)

    # Define inputs
    full_candidate = tf.keras.layers.Input(
        shape=(globals.data_params['article_size'] + globals.model_params['max_locations_per_article'] + 1),
        dtype="int32",
        name='candidate_Input',
    )  # 2d
    full_history = tf.keras.layers.Input(shape=(globals.model_params['max_no_documents_in_user_profile'],
                                                globals.data_params['article_size'] +
                                                globals.model_params['max_locations_per_article'] + 1),
                                         name='history_input',
                                         dtype="int32")

    # noinspection PyCallingNonCallable
    news_slicer = Slice(
        begin=0,
        size=globals.model_params['article_size'],
        name="slice_input_article"
    )
    # noinspection PyCallingNonCallable
    loc_slicer = Slice(
        begin=globals.model_params['article_size'],
        size=globals.data_params['max_locations_per_article'],
        name='slice_input_article_location_data',
    )
    pos_slicer = Slice(
        begin=globals.model_params['article_size'] + globals.data_params['max_locations_per_article'],
        size=1,
        name='slice_input_user_position_data',
    )
    # noinspection PyCallingNonCallable
    candidate = news_slicer(full_candidate)
    # noinspection PyCallingNonCallable
    candidate_loc = loc_slicer(full_candidate)
    # noinspection PyCallingNonCallable
    user_pos = pos_slicer(full_candidate)

    history = tf.keras.layers.TimeDistributed(news_slicer)(full_history)
    history_loc = tf.keras.layers.TimeDistributed(loc_slicer)(full_history)
    history_pos = tf.keras.layers.TimeDistributed(pos_slicer)(full_history)

    # Slice candidate and history and pop them into their encoders.
    # Run candidate through news encoder and history through user encoder
    encoded_candidate = news_encoder(candidate)
    encoded_user = user_encoder(history)

    encoded_cand_locs = candidate_location_encoder(candidate_loc)
    encoded_hist_locs = history_location_encoder(history_loc)

    encoded_user_pos = user_pos_encoder(user_pos)
    encoded_hist_pos = historical_pos_encoder(history_pos)
    encoded_comb_pos = user_hist_combined_pos_encoder([user_pos, history_pos])

    # Predict click probability # todo: outsource this to another file.
    news_score = tf.keras.layers.Dot(axes=-1)([encoded_candidate, encoded_user])  # Article interest score
    # loc_score = tf.keras.layers.Dot(axes=-1)([encoded_cand_locs, encoded_hist_locs])  # location interest score
    pos_score = tf.keras.layers.Dot(axes=-1)(
        [encoded_user_pos, encoded_cand_locs])  # news location to user position score
    hist_pos_score = tf.keras.layers.Dot(axes=-1)(
        [encoded_hist_pos, encoded_cand_locs])  # news location to hostorical user position score

    if historical:
        temp = [news_score, hist_pos_score]
    else:
        temp = [news_score, pos_score]

    if click_predictor == 'neural':
        neural_inputs = tf.concat(temp, axis=-1)
        x = tf.keras.layers.Dense(4, activation='relu')(neural_inputs)
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        full_model_output = x
    elif click_predictor == 'dot':  # this makes no sense to use. Makes it way harder for model to learn the thing.
        full_model_output = tf.keras.layers.Dot(axes=-1)(temp)
    else:
        raise NotImplementedError(f'Click predictor is implemented as Dot product ("dot") and neural network('
                                  f'"neural"). Could not find implementation for {click_predictor}.')

    # Define model & Compile
    model = tf.keras.models.Model([full_candidate, full_history], full_model_output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(globals.learning_params['learning_rate']),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.MeanSquaredError()],
    )

    return model


def task_to_params(task):
    """
    Returns the parameter from the task given on form ((loc, historical_loc), (pos, historical_pos))
    :param task:
    :return:
    """

    loc, historical_loc, pos, historical_pos = False, False, False, False

    task2params = {
        'news': ((False, False), (False, False)),
        'loc_to_loc': ((True, True), (False, False)),
        'pos_to_loc': ((True, False), (True, False)),
        'hist_pos_to_loc': ((True, False), (False, True)),
        'comb_pos_to_loc': ((True, False), (True, True)),
        'pos_loc_to_loc': ((True, True), (True, False)),
        'hist_pos_loc_to_loc': ((True, True), (False, True)),
        'comb_pos_loc_to_loc': ((True, True), (True, True)),
    }

    try:
        return task2params[task]
    except KeyError as e:
        raise KeyError(f"Task must be in the accepted tasks. Got {task}, but was not found in task2params.")


def naml_both(embedder: str = 'gensim', click_predictor='neural', task: str = 'news'):
    if task not in get_tasks():
        raise ValueError(f'Given task must be one of the predefined but was {task}. Check docs for get_tasks().')

    (loc, historical_loc), (pos, historical_pos) = task_to_params(task)

    # Build embedder, build news encoder based on embedder, build news encoder based on news encoder.
    if embedder == 'gensim':
        embedding_layer = get_embedding_layer_from_gensim()
    elif embedder == 'keras':
        embedding_layer = tf.keras.layers.Embedding(
            input_dim=globals.model_params['vocabulary_size'],
            output_dim=globals.model_params['word_embedding_dim'],
            embeddings_initializer="uniform",
            trainable=True,
            mask_zero=True,
        )
    else:
        raise ValueError(f'embedder parameter must be either "gensim" or "keras", but was {embedder}')

    news_encoder = build_news_encoder(embedding_layer=embedding_layer)
    user_encoder = define_user_encoder_model(
        input_shape=(globals.model_params['max_no_documents_in_user_profile'], globals.data_params['article_size']),
        encoding_operation=news_encoder
    )

    if loc or historical_loc:
        candidate_location_encoder, history_location_encoder = get_location_encoders(embedding_layer)

    if pos or historical_pos:
        user_pos_encoder, historical_pos_encoder, user_hist_combined_pos_encoder = get_position_encoders(
            embedding_layer)

    # Define inputs
    full_candidate = tf.keras.layers.Input(
        shape=(globals.data_params['article_size'] + globals.model_params['max_locations_per_article'] + 1,),
        # news + news locations + user position
        dtype="int32",
        name='full_candidate_input',
    )  #
    full_history = tf.keras.layers.Input(shape=(globals.model_params['max_no_documents_in_user_profile'],
                                                globals.data_params['article_size'] +
                                                globals.model_params['max_locations_per_article'] + 1),
                                         name='full_history_input',
                                         dtype="int32")

    # noinspection PyCallingNonCallable
    news_slicer = Slice(
        begin=0,
        size=globals.model_params['article_size'],
        name="slice_input_article"
    )
    if loc or historical_loc:
        loc_slicer = Slice(
            begin=globals.model_params['article_size'],
            size=globals.data_params['max_locations_per_article'],
            name='slice_input_article_location_data',
        )
    if pos or historical_pos:
        pos_slicer = Slice(
            begin=globals.model_params['article_size'] + globals.data_params['max_locations_per_article'],
            size=1,
            name='slice_input_user_position_data',
        )

    # noinspection PyCallingNonCallable
    candidate = news_slicer(full_candidate)
    history = tf.keras.layers.TimeDistributed(news_slicer)(full_history)

    # Run candidate through news encoder and history through user encoder
    encoded_candidate = news_encoder(candidate)
    encoded_user = user_encoder(history)

    if loc:
        # noinspection PyCallingNonCallable PyUnboundLocalVariable
        candidate_loc = loc_slicer(full_candidate)
        # noinspection PyUnboundLocalVariable
        encoded_cand_locs = candidate_location_encoder(candidate_loc)
    if historical_loc:
        history_loc = tf.keras.layers.TimeDistributed(loc_slicer)(full_history)
        # noinspection PyUnboundLocalVariable
        encoded_hist_locs = history_location_encoder(history_loc)

    if pos and historical_pos:
        # noinspection PyCallingNonCallable
        user_pos = pos_slicer(full_candidate)
        history_pos = tf.keras.layers.TimeDistributed(pos_slicer)(full_history)
        # noinspection PyUnboundLocalVariable
        encoded_comb_pos = user_hist_combined_pos_encoder([user_pos, history_pos])
    elif pos:
        # noinspection PyCallingNonCallable
        user_pos = pos_slicer(full_candidate)
        # noinspection PyUnboundLocalVariable
        encoded_user_pos = user_pos_encoder(user_pos)
    elif historical_pos:
        history_pos = tf.keras.layers.TimeDistributed(pos_slicer)(full_history)
        # noinspection PyUnboundLocalVariable
        encoded_hist_pos = historical_pos_encoder(history_pos)

    # Generating scores and preparing for combination # todo: outsource this to another file.
    similarity_function = None
    if similarity_function == 'dot':
        sim_func = tf.keras.layers.Dot(axes=-1)
    elif similarity_function == 'cosine':
        sim_func = tf.keras.layers.Dot(axes=-1, normalize=True)

    news_score = tf.keras.layers.Dot(axes=-1)([encoded_candidate, encoded_user])  # Article interest score
    if task == 'news':
        temp = [news_score]
    elif task == 'loc_to_loc':
        # noinspection PyUnboundLocalVariable
        loc_score = tf.keras.layers.Dot(axes=-1)([encoded_cand_locs, encoded_hist_locs])  # location interest score
        temp = [news_score, loc_score]
    elif task == 'pos_to_loc':
        # noinspection PyUnboundLocalVariable
        pos_score = tf.keras.layers.Dot(axes=-1)(
            [encoded_user_pos, encoded_cand_locs])  # news location to user position score
        temp = [news_score, pos_score]
    elif task == 'hist_pos_to_loc':
        # noinspection PyUnboundLocalVariable
        hist_pos_score = tf.keras.layers.Dot(axes=-1)(
            [encoded_hist_pos, encoded_cand_locs])  # news location to historical user position score
        temp = [news_score, hist_pos_score]
    elif task == 'comb_pos_to_loc':
        # noinspection PyUnboundLocalVariable
        comb_pos_score = tf.keras.layers.Dot(axes=-1)(
            [encoded_comb_pos, encoded_cand_locs])  # user current & historical position to news loc
        temp = [news_score, comb_pos_score]
    elif task == 'pos_loc_to_loc':
        # noinspection PyUnboundLocalVariable
        pos_score = tf.keras.layers.Dot(axes=-1)([encoded_user_pos, encoded_cand_locs])
        # noinspection PyUnboundLocalVariable
        loc_score = tf.keras.layers.Dot(axes=-1)([encoded_hist_locs, encoded_cand_locs])
        temp = [news_score, pos_score, loc_score]
    elif task == 'hist_pos_loc_to_loc':
        # noinspection PyUnboundLocalVariable
        hist_pos_score = tf.keras.layers.Dot(axes=-1)(
            [encoded_hist_pos, encoded_cand_locs])  # news location to historical user position score
        # noinspection PyUnboundLocalVariable
        loc_score = tf.keras.layers.Dot(axes=-1)([encoded_hist_locs, encoded_cand_locs])
        temp = [news_score, hist_pos_score, loc_score]
    elif task == 'comb_pos_loc_to_loc':
        # noinspection PyUnboundLocalVariable
        comb_pos_score = tf.keras.layers.Dot(axes=-1)(
            [encoded_comb_pos, encoded_cand_locs])  # user current & historical position to news loc
        # noinspection PyUnboundLocalVariable
        loc_score = tf.keras.layers.Dot(axes=-1)([encoded_hist_locs, encoded_cand_locs])
        temp = [news_score, comb_pos_score, loc_score]
    else:
        raise ValueError(f'Given task not accepted. Recieved {task}. See task_to_params for accepted values.')

    if len(temp) == 1:
        # If we only have a single similarity score, it does not need combining so it is selected for final usage.
        full_model_output = temp[0]
    elif len(temp) > 3:
        raise NotImplementedError('No methods have been implemented for tasks above 3 scorings.')

    elif click_predictor == 'dot':  # this makes no sense to use. Makes it way harder for model to learn the thing.
        # See model_location_adaptation.md for notes on XOR problem.
        if len(temp) == 2:
            full_model_output = tf.keras.layers.Dot(axes=-1)(temp)
        elif len(temp) == 3:
            x = tf.keras.layers.Dot(axes=-1)(temp[:-1])
            full_model_output = tf.keras.layers.Dot(axes=-1)([x, temp[-1]])
    elif click_predictor == 'neural':
        if len(temp) == 2:
            neural_inputs = tf.concat(temp, axis=-1)
            x = tf.keras.layers.Dense(4, activation='relu')(neural_inputs)
            x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
            full_model_output = x

        elif len(temp) == 3:
            neural_inputs = tf.concat(temp, axis=-1)
            x = tf.keras.layers.Dense(6, activation='relu')(neural_inputs)
            x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
            full_model_output = x
    elif click_predictor == 'sum':
        full_model_output = tf.reduce_sum(temp, axis=0)
    elif click_predictor == 'avg':
        full_model_output = tf.reduce_mean(temp, axis=0)
    elif click_predictor == 'max':
        full_model_output = tf.reduce_max(temp, axis=0)
    else:
        raise NotImplementedError(
            f'Click predictor is implemented as dot product ("dot"), sum("sum") and neural network("neural"). Could '
            f'not find implementation for {click_predictor} with scoring length {len(temp)}.')

    # Define model & Compile
    # noinspection PyUnboundLocalVariable
    model = tf.keras.models.Model([full_candidate, full_history], full_model_output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(globals.learning_params['learning_rate']),
        loss="binary_crossentropy",
        metrics=["accuracy", tfr.keras.metrics.get("ndcg", topn=5, name="NDCG@5")],
    )

    return model


def train_naml(model: tf.keras.models.Model, x_train, y_train, x_val, y_val, send_wandb: bool = True):
    callbacks = [tf.keras.callbacks.EarlyStopping('val_loss', patience=3, restore_best_weights=True)]
    if send_wandb:
        callbacks.append(WandbCallback())

    model.fit(x=x_train,
              y=y_train,
              batch_size=globals.learning_params['batch_size'],
              epochs=globals.learning_params['epochs'],
              validation_data=(x_val, y_val),
              callbacks=callbacks,
              )

    return model


def get_tasks():
    """
    Getter for the use cases passable to naml. This serves primarily as documentation for the code words.
    _________________________________________________
    news: Uses only the news articles themselves
    loc_to_loc: uses the historical article locations interests against the locations of the candidate
    pos_to_loc: uses the position of the user against the locations of the candidate
    hist_pos_to_loc: uses the historical position of the user against the locations of the candidate
    comb_pos_to_loc: uses the historical and current position of the user against the locations of the candidate
    pos_loc_to_loc: uses the historical location interests & position of the user against the locations of the candidate
    _________________________________________________

    :return: The accepted codewords for the naml task
    """
    return ['news', 'loc_to_loc', 'pos_to_loc', 'hist_pos_to_loc',
            'comb_pos_to_loc', 'pos_loc_to_loc', 'hist_pos_loc_to_loc', 'comb_pos_loc_to_loc']


if __name__ == '__main__':

    (x_c, x_u), y = data_generator.load_dataset_npy('locpos/20170101')
    cand_t, cand_v, hist_t, hist_v, y_t, y_v = sklearn.model_selection.train_test_split(x_c, x_u, y, random_state=
    globals.model_params['seed'])
    for task in get_tasks():
        wandb.init(
            project="NAML",
            name=f"{task}_neural_test",  # Should be something unique and informative... Defaults to strange texts that make no sense.
            # that is perfectly OK, since w & b will log all (hyper-)parameters anyway, and therefore connects name to data
            entity="adrianlangseth",
            notes="Just testing",
            job_type="train",
            config=globals.model_params,  # Adding all settings to have them logged on the w & b GUI
        )
        m = naml_both('keras', 'neural', task)
        train_naml(m, [cand_t, hist_t], y_t, [cand_v, hist_v], y_v, send_wandb=True)

        wandb.finish()

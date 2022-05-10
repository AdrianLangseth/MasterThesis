import numpy as np

from flexible_layers import FlexibleAveragePooling, MaskedAveragePooling
import tensorflow as tf
import globals


def build_location_encoder(embedding_layer: tf.keras.layers.Layer = None, multi:bool=True) -> tf.keras.models.Model:
    """
    The main function to create the news encoder of the NAML.
    news encoder in composed of title encoder, body encoder, vert encoder and subvert encoder
    Args:
        embedding_layer (keras Layer): a word embedding layer.
    Return:
        keras model: the news encoder of NAML.
    """

    # Fix input: If no word embedding is given, we make one
    if embedding_layer is None:
        embedding_layer = tf.keras.layers.Embedding(
            input_dim=globals.model_params['vocabulary_size'],
            output_dim=globals.model_params['word_embedding_dim'],
            embeddings_initializer="uniform",
            trainable=True,
        )

    location_inputs = tf.keras.layers.Input(
        shape=(
            globals.data_params['max_locations_per_article'],
        ), dtype="int32"
    )

    embedded_locations = embedding_layer(location_inputs)

    reshaped = tf.keras.layers.Reshape((-1, embedded_locations.shape[-1]))(embedded_locations)  #  ,globals.data_params['location_embedding_vector_size']

    transposed = tf.transpose(reshaped, perm=[0, 2, 1])

    # noinspection PyCallingNonCallable
    averaged = FlexibleAveragePooling()(transposed)

    # Define everything as a Keras model, taking the input as defined and returning the attended
    # concatenated representation
    model = tf.keras.models.Model(location_inputs, averaged, name="location_encoder")
    return model


def get_location_encoders(embedding_layer):

    def cand_loc(embedding_layer):
        inputs = tf.keras.layers.Input(
            shape=(
                globals.data_params['max_locations_per_article'],
            ), dtype='int32'
        )  # (batch, max_loc)

        x = embedding_layer(inputs) # (batch, max_loc, embed)
        x = MaskedAveragePooling()(x)
        return tf.keras.models.Model(inputs, x, name='candidate_location_encoder')

    def hist_loc(embedding_layer):
        inputs = tf.keras.layers.Input(
            shape=(
                globals.data_params['max_no_documents_in_user_profile'],
                globals.data_params['max_locations_per_article']
            ), dtype='int32'
        )  # (batch, hist, max_loc)

        x = embedding_layer(inputs)  # (batch, hist, max_loc, embed)
        x = tf.keras.layers.Reshape((-1, embedding_layer.output_dim))(x)  # (batch, hist*max_loc, embed)
        x = MaskedAveragePooling()(x)

        return tf.keras.models.Model(inputs, x, name='historical_location_encoder')

    return cand_loc(embedding_layer), hist_loc(embedding_layer)


def get_old_location_encoders(embedding_layer):

    def cand_loc(embedding_layer):
        inputs = tf.keras.layers.Input(
            shape=(
                globals.data_params['max_locations_per_article'],
            ), dtype='int32'
        )  # (batch, max_loc)

        x = embedding_layer(inputs) # (batch, max_loc, embed)
        x = tf.transpose(x, perm=[0, 2, 1])
        # noinspection PyCallingNonCallable
        x = FlexibleAveragePooling()(x)
        return tf.keras.models.Model(inputs, x, name='candidate_location_encoder')

    def hist_loc(embedding_layer):
        inputs = tf.keras.layers.Input(
            shape=(
                globals.data_params['max_no_documents_in_user_profile'],
                globals.data_params['max_locations_per_article']
            ), dtype='int32'
        )  # (batch, hist, max_loc)

        x = embedding_layer(inputs)  # (batch, hist, max_loc, embed)
        x = tf.keras.layers.Reshape((-1, embedding_layer.output_dim))(x)  # (batch, hist*max_loc, embed)
        x = tf.transpose(x, perm=[0, 2, 1])
        # noinspection PyCallingNonCallable
        x = FlexibleAveragePooling()(x)

        return tf.keras.models.Model(inputs, x)

    return cand_loc(embedding_layer), hist_loc(embedding_layer)

def get_old_location_encoders(embedding_layer):

    def cand_loc(embedding_layer):
        inputs = tf.keras.layers.Input(
            shape=(
                globals.data_params['max_locations_per_article'],
            ), dtype='int32'
        )  # (batch, max_loc)

        x = embedding_layer(inputs) # (batch, max_loc, embed)
        x = tf.transpose(x, perm=[0, 2, 1])
        # noinspection PyCallingNonCallable
        x = FlexibleAveragePooling()(x)
        return tf.keras.models.Model(inputs, x, name='candidate_location_encoder')

    def hist_loc(embedding_layer):
        inputs = tf.keras.layers.Input(
            shape=(
                globals.data_params['max_no_documents_in_user_profile'],
                globals.data_params['max_locations_per_article']
            ), dtype='int32'
        )  # (batch, hist, max_loc)

        x = embedding_layer(inputs)  # (batch, hist, max_loc, embed)
        x = tf.keras.layers.Reshape((-1, embedding_layer.output_dim))(x)  # (batch, hist*max_loc, embed)
        x = tf.transpose(x, perm=[0, 2, 1])
        # noinspection PyCallingNonCallable
        x = FlexibleAveragePooling()(x)

        return tf.keras.models.Model(inputs, x)

    return cand_loc(embedding_layer), hist_loc(embedding_layer)


def get_position_encoders(embedding_layer):

    def user_pos(embedding_layer):
        user_pos_inputs = tf.keras.layers.Input(
            shape=(1,), name='user_position_input' , dtype='int32',
        )  # shape (batch_size, 1)
        x = embedding_layer(user_pos_inputs)  # shape (batch_size, 1, embed_dim)
        x = tf.squeeze(x, axis=-2)   # shape (batch_size, embed_dim)

        return tf.keras.models.Model(user_pos_inputs, x, name='user_position_encoder')

    def hist_pos(embedding_layer):
        hist_inputs = tf.keras.layers.Input(
            shape=(
                globals.data_params['max_no_documents_in_user_profile'], 1
            ), dtype='int32', name='historical_user_position_input'
        )  # Will come in on shape (batch_size, hist_size, 1)

        x = embedding_layer(hist_inputs)  # (batch_size, hist_size, 1, embed_dim)
        x = tf.squeeze(x, axis=-2)  # (batch_size, hist_size, embed_dim)
        x = tf.transpose(x, perm=[0, 2, 1])  # (batch_size, embed_dim, hist_size)
        # noinspection PyCallingNonCallable
        x = FlexibleAveragePooling()(x)  # (batch_size, embed_dim)

        return tf.keras.models.Model(hist_inputs, x, name='historical_user_position_encoder')

    def hist_user_pos(embedding_layer):
        """
        This one includes the current position so the news location os matched to the user position in the past and the
        one right now.
        :param embedding_layer:
        :return:
        """
        user_pos_inputs = tf.keras.layers.Input(
            shape=(
                1,
            ), name='current_user_position_input', dtype='int32'
        )  # shape (batch_size, 1)
        hist_inputs = tf.keras.layers.Input(
            shape=(
                globals.data_params['max_no_documents_in_user_profile'], 1
            ), dtype='int32', name='historical_user_position_input'
        )  # shape (batch_size, hist_size, 1)

        x = tf.concat([tf.expand_dims(user_pos_inputs, axis=-2), hist_inputs], axis=-2)
        # shape (batch_size, hist_size + 1, 1)

        x = embedding_layer(x)  # (batch_size, hist_size + 1, 1, embed_dim)
        x = tf.squeeze(x, axis=-2)  # (batch_size, hist_size + 1, embed_dim)
        x = tf.transpose(x, perm=[0, 2, 1])  # (batch_size, embed_dim, hist_size + 1)
        # noinspection PyCallingNonCallable
        x = FlexibleAveragePooling()(x)  # (batch_size, embed_dim)

        return tf.keras.models.Model([user_pos_inputs, hist_inputs], x, name='past_and_current_user_position_encoder')


    return user_pos(embedding_layer), hist_pos(embedding_layer), hist_user_pos(embedding_layer)



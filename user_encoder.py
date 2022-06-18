import tokenizer_embedder
from flexible_layers import ChineseAttention
import tensorflow as tf
import data_generators
import globals


def define_user_encoder_model_wo_mask(input_shape: tuple, encoding_operation: tf.keras.models.Model, ) -> tf.keras.models.Model:
    """
    Define a model that works as follows:
    -- It gets as input a number of batches, which is the first dim of the input.
    -- Each record is a collection of documents. The documents are ids for text that will be encoded
    So the input is a 3d tensor where the dimensions are
    -- user
    -- document
    -- word
    input[i, j, k] is therefore user <i> in the data-batch, document <j> for this user, id for word <k> for that user.

    The result is an encoded "user-description"

    NOTE!!
    The encoding operation (a Keras model that takes a document as a sequence of word IDs and returns
    an embedding of the document) is defined **externally** and is an input to this part.
    The reason is that the encoder typically will rely on some embeddings. If we define the model with embeddings
    inside this function we will not get the same embeddings as we do elsewhere (because they are defined locally).
    Solution is to define the word-embedder once and for all, so that the embedding-solutions shared globally.
    Recommended practice here is obviously to use build_news_encoder() from news_encoder.py as the
    document encoding thing


    Args:
        input_shape: Tuple describing the input-shape
        encoding_operation (tf.keras.models.Model): The encoder of the document. Supposed to be a Keras model

    Return:
        tf.keras.models.Model: a Keras model that defines the user-model
    """

    """
    Define input-layer. Note that data-type is int32, so that it fits with the data-assumption above.
    These are word-indexes that will be given to the embedder
    """
    input_layer = tf.keras.layers.Input(shape=input_shape, dtype="int32")

    """
    Do a "time-distributed" operation. What this means is simply that we slice the 
    input through the second dim (document-id) and do the same operation for each.
    That can for instance be to run through an encoder layer.

    There is an extra dim of size 1 introduced by the TimeDistributed layer at position 2 that must be removed,
    so that's the point of the if statement. 
    """
    encoded = tf.keras.layers.TimeDistributed(encoding_operation)(input_layer)
    if len(encoded.shape) == 4:
        encoded = tf.squeeze(encoded, axis=2)

    """
    Now do attention on that stuff on the encoded data. 
    Note that the syntax here is correct, even though pylint doesn't understand it 
    """
    # noinspection PyCallingNonCallable
    output_layer = ChineseAttention(dim=globals.model_params['attention_hidden_dim'],
                                    seed=globals.learning_params['seed'])(encoded)

    # We are done: Define the model and leave
    model = tf.keras.models.Model(input_layer, output_layer, name="user_encoding")
    return model


def define_user_encoder_model(input_shape: tuple, encoding_operation: tf.keras.models.Model, ) -> tf.keras.models.Model:
    """
    Define a model that works as follows:
    -- It gets as input a number of batches, which is the first dim of the input.
    -- Each record is a collection of documents. The documents are ids for text that will be encoded
    So the input is a 3d tensor where the dimensions are
    -- user
    -- document
    -- word
    input[i, j, k] is therefore user <i> in the data-batch, document <j> for this user, id for word <k> for that user.

    The result is an encoded "user-description"

    NOTE!!
    The encoding operation (a Keras model that takes a document as a sequence of word IDs and returns
    an embedding of the document) is defined **externally** and is an input to this part.
    The reason is that the encoder typically will rely on some embeddings. If we define the model with embeddings
    inside this function we will not get the same embeddings as we do elsewhere (because they are defined locally).
    Solution is to define the word-embedder once and for all, so that the embedding-solutions shared globally.
    Recommended practice here is obviously to use build_news_encoder() from news_encoder.py as the
    document encoding thing


    Args:
        input_shape: Tuple describing the input-shape
        encoding_operation (tf.keras.models.Model): The encoder of the document. Supposed to be a Keras model

    Return:
        tf.keras.models.Model: a Keras model that defines the user-model
    """

    """
    Define input-layer. Note that data-type is int32, so that it fits with the data-assumption above.
    These are word-indexes that will be given to the embedder
    """
    input_layer = tf.keras.layers.Input(shape=input_shape, dtype="int32")
    "Shape: (batch, hist_size, 73)"
    " We calculate a mask of shape (batch, hist_size)"
    mask = tf.reduce_any(input_layer != 0, axis=-1)


    """
    Do a "time-distributed" operation. What this means is simply that we slice the 
    input through the second dim (document-id) and do the same operation for each.
    That can for instance be to run through an encoder layer.

    There is an extra dim of size 1 introduced by the TimeDistributed layer at position 2 that must be removed,
    so that's the point of the if statement. 
    """
    encoded = tf.keras.layers.TimeDistributed(encoding_operation)(input_layer)
    if len(encoded.shape) == 4:
        encoded = tf.squeeze(encoded, axis=2)

    """
    Now do attention on that stuff on the encoded data. 
    Note that the syntax here is correct, even though pylint doesn't understand it 
    """
    # noinspection PyCallingNonCallable
    output_layer = ChineseAttention(dim=globals.model_params['attention_hidden_dim'],
                                    seed=globals.learning_params['seed'])(encoded, mask=mask)

    # We are done: Define the model and leave
    model = tf.keras.models.Model(input_layer, output_layer, name="user_encoding")
    return model


def __test_user_encoder_learning() -> tf.keras.models.Model:
    """
    Testing that we can make a model that learns something

    Args:
    return: The trained model
    """

    """
    Data generator. Gives a 3d tensor of shape [no_users, no_docs, total-length-of-dic-repr]. 
    """
    training_data = data_generators.user_doc_data_generator(no_users=50000,
                                                            also_generate_class=True)

    val_data = data_generators.user_doc_data_generator(no_users=1000,
                                                       also_generate_class=True)
    input_shape = tuple(training_data['x'].shape[1:])
    """
    Define the embedder.
    """
    emb = tokenizer_embedder.get_embedding_layer_from_gensim()

    shared_word_embedding_layer = tf.keras.layers.Embedding(
        input_dim=globals.model_params['vocabulary_size'],
        output_dim=globals.model_params['word_embedding_dim'],
        embeddings_initializer="uniform",
        trainable=True,
    )
    # Doing the dirty here: Import in the middle of the coder. I don't want this part to be loaded unless I have to
    from news_encoder import build_news_encoder
    embedder_layer = build_news_encoder(embedding_layer=emb) # shared_word_embedding_layer)

    """
    Define the user-embedding: It will take the documents from the user and make some embedding
    """
    user_embedding = define_user_encoder_model(input_shape=input_shape,
                                               encoding_operation=embedder_layer)

    full_model_input = tf.keras.layers.Input(shape=input_shape, dtype="int32")
    # noinspection PyCallingNonCallable
    attended = user_embedding(full_model_input)
    full_model_output = tf.keras.layers.Dense(1, activation="sigmoid")(attended)
    model = tf.keras.models.Model(full_model_input, full_model_output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(globals.learning_params['learning_rate']),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(x=training_data['x'],
              y=training_data['y'],
              batch_size=globals.learning_params['batch_size'],
              epochs=globals.learning_params['epochs'],
              validation_data=(val_data['x'], val_data['y']),
              #callbacks=[WandbCallback()],
              )

    return model

if __name__ == "__main__":
    __test_user_encoder_learning().save("./saved_model")

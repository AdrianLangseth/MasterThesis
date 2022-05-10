import tensorflow as tf
from flexible_layers import ChineseAttention, Slice
import data_generators
import globals


def _build_text_encoder_model(input_shape: tuple,
                              embedding_layer: tf.keras.layers.Layer,
                              last_name: str
                              ) -> tf.keras.models.Model:
    """
    This thing builds the encoder for the title or body of the article and
    runs a number of layers:
    - Embedding
    - Dropout
    - 1D convolution
    - Dropout again
    - Chinese attention
    - Reshape to get the correct representation shape

    Everything is wrapped in a keras model, so we can call the result later on

    Args:
        input_shape (tuple): the shape of the input. Will be, e.g.,  (globals.model_params['body_size'],)
        embedding_layer (keras layer): a word embedding layer. Shared across all such operations to ensure
        that it is the same embedding layer used everywhere
        last_name (str): added to the model's name so that it is unique, and we can find it in the graph
    Return:
        keras model: Used to define the representation of a document part
    """

    input_layer = tf.keras.layers.Input(shape=input_shape, dtype="int32")

    # Embedding
    result = embedding_layer(input_layer)

    # Dropout
    result = tf.keras.layers.Dropout(globals.model_params['dropout_probability'])(result)

    # Convolution
    result = tf.keras.layers.Conv1D(
        globals.model_params['no_conv_filters'],
        globals.model_params['conv_window_size'],
        activation=globals.model_params['conv_activation_function'],
        padding='same',
        bias_initializer=tf.keras.initializers.Zeros(),
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=globals.learning_params['seed']),
    )(result)

    # Dropout again
    result = tf.keras.layers.Dropout(globals.model_params['dropout_probability'])(result)

    # Chinese attention.
    # noinspection PyCallingNonCallable
    result = ChineseAttention(globals.model_params['attention_hidden_dim'],
                              seed=globals.learning_params['seed']
                              )(result)

    # Reshape
    result = tf.keras.layers.Reshape((1, globals.model_params['no_conv_filters']))(result)

    # ... and we are done: Just send the model back
    model = tf.keras.models.Model(input_layer, result, name="text_string_encode_" + last_name)
    return model


def _build_category_encoder_model(no_possible_choices: int, last_name: str) -> tf.keras.models.Model:
    """
    Create a keras Model that takes a (sub) category, and returns an embedding.
    Same stuff for main category and subcategory, so no need for two methods as in the original code
    :param no_possible_choices: The number of categories for this data (i.e., nof categories or no subcategories)
    :param last_name: string added to model name, so we find it in the computational graph
    :return: A keras model
    """

    # Define input-layer
    input_layer = tf.keras.layers.Input(shape=(1,), dtype="int32")

    # Make embedding layer and run it
    category_embedding_layer = tf.keras.layers.Embedding(
        no_possible_choices,    # Possible values taken
        globals.model_params['category_embedding_dim'],     # Embedding size
        trainable=True
    )
    result = category_embedding_layer(input_layer)

    # Make a dense to get correct dimensions
    result = tf.keras.layers.Dense(
        globals.model_params['no_conv_filters'],    # Number of outputs set to the same as for the text convolutions
        activation=globals.model_params['dense_activation_function'],
        bias_initializer=tf.keras.initializers.Zeros(),
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=globals.learning_params['seed']),
    )(result)
    result = tf.keras.layers.Reshape((1, globals.model_params['no_conv_filters']))(result)

    # Define the result as a model and leave
    model = tf.keras.models.Model(input_layer, result, name="category_encoder_" + last_name)
    return model


def build_news_encoder(embedding_layer: tf.keras.layers.Layer = None) -> tf.keras.models.Model:
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

    # Create input for the document info.
    # Data is integers, and consists of:
    #  -- document title text (word IDs, max length globals.model_params['title_size'])
    #  -- document body text (word IDs, max length globals.model_params['body_size'])
    #  -- Some category info. Still represented by integers, and dim is 2: Toplevel category, sub-category
    # Note that the data is sent to us in "one go": shape is (batch_size, title-size+body-size+2)
    input_title_body_categories = tf.keras.layers.Input(
        shape=(
            globals.model_params['title_size'] +
            globals.model_params['body_size'] +
            2,      # One for main category, one for sub-category. Don't know what these are, but taken from original
        ), dtype="int32"
    )

    # Now we divide the input into its separate parts: Title, Body, main category, sub-category
    # Original code used Lambda-layers, which is a mess, so re-implemented using a new Slice-layer in
    # flexible_layers.py
    # Unfortunately pylint thinks the call to the layer is illegal, so Pycharm makes a mess, but it is correct
    # noinspection PyCallingNonCallable
    input_only_title = Slice(
        begin=0,
        size=globals.model_params['title_size'],
        name="input_only_title"
    )(input_title_body_categories,)
    # noinspection PyCallingNonCallable
    input_only_body = Slice(
        begin=globals.model_params['title_size'],
        size=globals.model_params['body_size'],
        name="input_only_body"
    )(input_title_body_categories)
    # noinspection PyCallingNonCallable
    input_main_category = Slice(
        begin=globals.model_params['title_size'] + globals.model_params['body_size'],
        size=1,
        name='input_main_category',
    )(input_title_body_categories)
    # noinspection PyCallingNonCallable
    input_sub_category = Slice(
        begin=globals.model_params['title_size'] + globals.model_params['body_size'] + 1,
        size=1,
        name='input_sub_category',
    )(input_title_body_categories)

    # Now we have the different input-types separated, and can run the operations on each of them
    # The original code had separate methods for title and body, but that is not needed.
    # Input shape for each call is given by the length of the text-object (title or body).
    # Output-shape for each of these four calls will be (1, globals.model_params['no_conv_filters']),
    # meaning that globals.model_params['no_conv_filters'] is the encoding dim we are applying here
    title_representation = _build_text_encoder_model(
        input_shape=(globals.model_params['title_size'],),
        embedding_layer=embedding_layer,
        last_name="title",
    )(input_only_title)

    body_representation = _build_text_encoder_model(
        input_shape=(globals.model_params['body_size'],),
        embedding_layer=embedding_layer,
        last_name="body",
    )(input_only_body)

    # Encode categories. Again two methods in the original code are merged into one.
    category_representation = _build_category_encoder_model(
        globals.model_params['no_categories'],
        last_name="main",
    )(input_main_category)
    sub_category_representation = _build_category_encoder_model(
        globals.model_params['no_sub_categories'],
        last_name="sub",
    )(input_sub_category)

    # Each input type has been handled separately. Let's merge the results together.
    # This is simply done by concatenating the results. Axis = -2 is in the original code
    # The resulting structure has size (batch-size, 4, encoding-dim).
    # The 4 is due to us looking at 4 "things": Title, body, main-category, sub-category
    # We will next use Chinese attention over the four data-sources
    concatenated_representation = tf.keras.layers.Concatenate(axis=-2)(
        [title_representation, body_representation, category_representation, sub_category_representation]
    )

    # Use attention on the concatenated representations
    # noinspection PyCallingNonCallable
    news_representation = ChineseAttention(
        globals.model_params['attention_hidden_dim'],
        seed=globals.learning_params['seed']
    )(concatenated_representation)

    # Define everything as a Keras model, taking the input as defined and returning the attended
    # concatenated representation
    model = tf.keras.models.Model(input_title_body_categories, news_representation, name="news_encoder")
    return model


def __test_news_encoder_forward_pass():
    """
    Testing that the forward pass works. There is no learning, just checking that data can be
    pushed through without crashing.

    return: A model (that I hope works)
    """

    """
    Data generator. Gives a 2d tensor of shape [no_documents, document-representation-size]. 
    Each entry apart from last two columns are in [0, ... vocab_size - 1], with both end-points being possible
    Last two columns encode category and subcategory, and max-values are chose respectively
    """
    data = data_generators.document_data_generator()

    """
    Define the embedder. This is a simple embedding-layer from Keras. Defined "externally" to makes sure the same 
    weights are used all over  
    """
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=globals.model_params['vocabulary_size'],
        output_dim=globals.model_params['word_embedding_dim'],
        embeddings_initializer='uniform',
        trainable=True,
    )

    model = build_news_encoder(embedding_layer=embedding_layer)
    model.summary()

    # noinspection PyCallingNonCallable
    result = model(data)

    print(f"Result shape was {result.shape}; should be "
          f"({globals.model_params['max_no_documents_in_user_profile']}, "
          f"{globals.model_params['no_conv_filters']})")

    return model


def __test_news_encoder_learning():
    """
    Testing that the learning works.

    return: A model (that I hope works)
    """

    """
    Data generator. Gives a 2d tensor of shape [no_docs, document-representation-size]. 
    Each entry apart from last two columns are in [0, ... vocab_size - 1], with both end-points being possible
    Last two columns encode category and subcategory, and max-values are chose respectively
    """
    training_data = data_generators.document_data_generator(also_generate_class=True)
    val_data = data_generators.document_data_generator(also_generate_class=True)

    """
    Define the embedder. This is a simple embedding-layer from Keras. 
    Defined "externally" in this test-function to makes sure the same 
    weights are used all over  
    """
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=globals.model_params['vocabulary_size'],
        output_dim=globals.model_params['word_embedding_dim'],
        embeddings_initializer='uniform',
        trainable=True,
    )
    news_model = build_news_encoder(embedding_layer=embedding_layer)

    # Define the full classification model
    full_model_input = tf.keras.layers.Input(
        shape=(globals.model_params['title_size'] + globals.model_params['body_size'] + 2),
        dtype="int32"
    )
    # noinspection PyCallingNonCallable
    encoded = news_model(full_model_input)
    full_model_output = tf.keras.layers.Dense(1, activation="sigmoid")(encoded)
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
              # callbacks=[WandbCallback()],
              )
    return model


if __name__ == "__main__":
    # test_forward_pass()
    __test_news_encoder_learning()

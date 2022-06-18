import tensorflow as tf
import numpy as np


class FlexibleAveragePooling(tf.keras.layers.Layer):
    """
    Average pooling over a tensor of shape sized at least 3:
    As always, the first is for the batch, i.e., the number of observations processed at the same time.
    The last dim is the one we average over.
    Then, we have at least one dimension for the data we are considering in-between
    So, we understand the dims as the first one being the index of the batch, the last dim is the indexing of
    data-objects related to that observation, and the dims inbetween are the actual data.
    We want to average over the data-objects, i.e., the last dim.
    Hence, input shape to the layer is  [batch-size, <whatever>, size-dim-to-average-over]
    and output shape is [batch-size, <whatever>].

    The reason we need a new implementation of this and not use Keras implementation of AveragePooling
    is that we want the flexibility to accommodate the situation where some data is "empty".
    An object is empty is all elements in [<batch-id>, :, :, <object-id>] == 0
    """

    def __init__(self):
        """
        Nothing special going on here -> Pass to super
        """
        super(FlexibleAveragePooling, self).__init__()

    # noinspection PyMethodMayBeStatic
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Do the operation: Average over the layers (given by last din) that are not zero
        :param inputs: input to layer: shape (None, ..., k)
        :return: output from layer. Shape same as input but dropping last dim: (None, ...)
        """
        # max_values holds the max (in absolute value) over all <whatever>-dims per <batch-id>, <object-id>
        # The size is therefore [batch-size, size-dim-to-average-over]
        max_values = tf.reduce_max(tf.abs(inputs), axis=range(1, len(inputs.shape) - 1))

        # no_legal_inputs counts the number of non-zero data-dims per record.
        # To avoid numerical issues in settings where we have high data-dims, the
        # check is not if the max is > 0, but if it is > some small number.
        # Shape is [batch-size]
        no_legal_inputs = tf.math.reduce_sum(tf.cast(tf.math.greater(max_values, 1E-8), tf.int32), axis=-1)

        # We add a small positive constant to no_legal_inputs to avoid a 0/0 issue if data
        # contains no information at all (now, the result will be a zero-tensor
        # Shape remains [batch-size]
        no_legal_inputs = tf.cast(no_legal_inputs, tf.float32) + tf.constant(1e-8)

        # summed_values holds the summed values over data-dims
        # shape is [batch-size, <whatever>]
        summed_values = tf.reduce_sum(inputs, keepdims=False, axis=-1)

        # Reshape no_legal_inputs that the vector now is of shape[batch-size, 1, 1, ..., 1],
        # meaning that its number of dims equals the number of dims in summed_values
        desired_dim = np.array(summed_values.shape)
        desired_dim[1:] = 1
        desired_dim[0] = -1
        no_legal_inputs = tf.reshape(no_legal_inputs, desired_dim)

        # Division: Returns structure with shape [batch-size, <whatever>]
        outputs = summed_values / no_legal_inputs

        # Done
        return outputs

    # noinspection PyMethodMayBeStatic
    def compute_output_shape(self, child_input_shape):
        """
        Have to help Keras to understand the output shape.
        It will be the shape of the part "cut out"
        :param child_input_shape: Output-shape of the layer below
        :return: Output-shape: Same as input, but dropping last dim
        """
        return child_input_shape[:-1]


class MaskedAveragePooling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MaskedAveragePooling, self).__init__(**kwargs)

    # noinspection PyMethodMayBeStatic
    def compute_mask(self, input, input_mask=None):
        # We do not need to pass the mask in our use case.

        if input_mask is None:
            return None

        float_mask = tf.cast(input_mask, tf.float32)
        embed_sized_float_mask = tf.keras.backend.repeat(float_mask, input.shape[-1])
        shaped_embed_sized_float_mask = tf.transpose(embed_sized_float_mask, [0, 2, 1])

        return shaped_embed_sized_float_mask

    # noinspection PyMethodMayBeStatic
    def call(self, x, mask=None):
        if mask is None:
            return tf.reduce_mean(x, axis=-2)

        float_mask = tf.cast(mask, tf.float32)
        embed_sized_float_mask = tf.keras.backend.repeat(float_mask, x.shape[-1])
        shaped_embed_sized_float_mask = tf.transpose(embed_sized_float_mask, [0, 2, 1])
        corrected_embeds = x * shaped_embed_sized_float_mask
        return tf.reduce_sum(corrected_embeds, axis=-2) / (1.E-8 + tf.reduce_sum(shaped_embed_sized_float_mask, axis=-2))

    # noinspection PyMethodMayBeStatic
    def compute_output_shape(self, input_shape):
        """
        :param input_shape: ( batch, locs, embed_size)
        :return: (batch, embed_size)
        """
        return (input_shape[0], input_shape[2])


class DocumentEmbedder(tf.keras.layers.Layer):
    """
    This is just a simple fix to have a document encoder defined as a layer.
    It was used early on as a debug-tool. Later replaced by the news_encoder model
    that is defined in
    """

    def __init__(self, words_per_doc: int, vocab_size: int, embedding_size: int, **kwargs):
        """
        This is procedure to take a single document and find some vector-representation of it
        :param words_per_doc: Words in the document
        :param vocab_size: Words in the vocabulary
        :param embedding_size: Dim of vector-space
        :return: A Keras model that takes a document in the form
            of a list of integers, and returns vector of size <embedding_size>
        """
        self.words_per_doc = words_per_doc
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding_layer = None
        super(DocumentEmbedder, self).__init__(**kwargs)

    def get_config(self):
        """ This is to fix so that the model can be saved"""
        config = super().get_config().copy()
        config.update({
            "words_per_doc": self.words_per_doc,
            "vocab_size": self.vocab_size,
            "embedding_size": self.embedding_size,
        })
        return config

    def build(self, input_shape: tf.TensorShape):
        if len(input_shape) != 2:
            raise ValueError("DocumentEmbedder wants 2d input:\n "
                             "- Documents\n"
                             "- Words in each document\n"
                             f"Now the input was {len(input_shape)}-d, with shape "
                             f"{input_shape}")
        if input_shape[-1] != self.words_per_doc:
            raise ValueError("DocumentEmbedder wants 2d input, and the 2nd dim "
                             "is supposed to be the no. words per document. \n"
                             "When the layer was defined this parameter was set to"
                             f"{self.words_per_doc}, but the input is supposed to be"
                             f"{input_shape[-1]}.")

        """
        Note two things here: 
            1) I am using embeddings_initializer="uniform" and trainable = True.  One could of course send in the 
                gensim vectors instead. This is just to test that stuff works, so couldn't be bothered now
            2) There is an optional parameter input_length that can be sent in. This is the length of each document. 
                I don't think we need it here, but if we do it is just input_length=self.words_per_doc...  
        """
        self.embedding_layer = tf.keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer="uniform",
            trainable=True,
            # input_length=self.words_per_doc,
        )
        super(DocumentEmbedder, self).build(input_shape)

    def call(self, inputs):
        """
        Call operation: Generate embeddings of all words in the document, then
        simply average over them. This is of course a crappy setup, but I am just testing here
        :param inputs: Input-tensor to the layer. These are word IDs, so integers.
            Shape is (None, self.words_per_doc)
        :return: The result of the averaging after lookup of vectors
        """
        output = self.embedding_layer(inputs)
        output = tf.reduce_mean(output, axis=-2, keepdims=False)
        return output

    def compute_output_shape(self, child_input_shape):
        """
        Have to help Keras to understand the output shape.
        It will be a vector of length  self.embedding_size per document
        :param child_input_shape: Output-shape of the layer below
        :return: Output-shape: (batch-size, embedding-size)
        """
        return child_input_shape[0], self.embedding_size


class ChineseAttention(tf.keras.layers.Layer):
    """
    Soft alignment attention implement.
    Starting form AttLayer2 in NAML, see
    https://github.com/microsoft/recommenders/blob/d4181cf1d1df6e71f7e6b202b0875bb3bd54150c/recommenders/models/newsrec/models/layers.py

    Changes here: Clean up code, and make it run under Tensorflow v2

    Attributes:
        dim (int): attention hidden dim
    """

    def __init__(self, dim=200, seed=0, **kwargs):
        """Initialization steps for AttLayer2.
        Args:
            dim (int): attention hidden dim
        """

        # Remember inputs
        self.dim = dim
        self.seed = seed

        # Define other stuff that will be used in the building of the layer@
        self.W, self.b, self.q = None, None, None

        # Call super's init
        super(ChineseAttention, self).__init__(**kwargs)

    def build(self, input_shape: tf.TensorShape):
        """
        Initialization for variables in ChineseAttention
        Trainable variables in the layer are W, b and q.

        Args:
            input_shape (object): shape of input tensor.

        """

        if len(input_shape) != 3:
            raise ValueError(f"ChineseAttention only handles 3 dim data-inputs. "
                             f"Now it got input-shape of length {len(input_shape)}, "
                             f"specifically shape was {input_shape}")

        # Add the trainable parameters: W, b, q
        # We do so using "add_weight", a method defined at the superclass (tf.keras.layers.Layer)
        self.W = self.add_weight(
            name="W",
            shape=(int(input_shape[-1]), self.dim),
            initializer=tf.keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True,
        )
        self.b = self.add_weight(
            name="b",
            shape=(self.dim,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )
        self.q = self.add_weight(
            name="q",
            shape=(self.dim, 1),
            initializer=tf.keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True,
        )

        # Trainable params are ready -- let's pass on to super's build method
        super(ChineseAttention, self).build(input_shape)  # be sure you call this somewhere!

    def get_config(self):
        """ This is to fix so that the model can be saved"""
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "seed": self.seed,
        })
        return config

    def call(self, inputs, mask=None):
        """
        Core implementation of soft attention
        Args:
            inputs (object): input tensor.
            mask: potential masking of some elements form the attention.
            Used e.g. in time-series if we don't want the choice made at time t to get
            information about whatever happens at time t+1

        Returns:
            object: weighted sum of input tensors.
        """

        """
        Step 1: Do element-wise tanh of (input * W + b)
        Shapes: 
           input: A tensor with three dims
           W: [input_shape[-1], self.dim]
           b: [self.dim,]
        Original code: attention = K.tanh(K.dot(inputs, self.W) + self.b)
        K.dot(inputs, self.W) is a tensor-dot operation over the last dim of inputs vs the first of W, 
            resulting in a shape of size [input_shape[0], input_shape[1], self.dim]
        Adding self.b, of shape [self.dim,] will broadcast and leave the shape unchanged    
        """
        attention = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)

        """
        Step 2: Multiply with the query
        Shapes: 
           attention: A tensor with shape [input_shape[0], input_shape[1], self.dim]
           q: [self.dim, 1]
        Original code: attention = K.dot(attention, self.q)
        K.dot(attention, self.q) is a tensor-dot operation over the last dim of attention vs the first of q, 
            resulting in a shape of size [input_shape[0], input_shape[1], 1]    
        """
        attention = tf.tensordot(attention, self.q, axes=1)

        """
        Step 3: Squeeze out the last dim of the representation, so 
        going from a tensor of shape  [input_shape[0], input_shape[1], 1] 
        down to [input_shape[0], input_shape[1]]
        
        Original code: attention = K.squeeze(attention, axis=2)
        """
        attention = tf.squeeze(attention, axis=-1)

        """
        Step 4: 
        The attentions are going to go through the softmax. If there is a mask, these elements will be 
        cancelled out. First step here is to take the exponential (only change from original code is to 
        swap K.exp and K.cast with their tf. equivalents 
        """
        if mask is None:
            attention = tf.exp(attention)
        else:
            attention = tf.exp(attention) * tf.cast(mask, dtype="float32")

        """
        Step 5: Do the weighing to get a distribution that sums to 1.
        Original code:
        attention_weight = attention / (K.sum(attention, axis=-1, keepdims=True) + K.epsilon())
        Changed K. to tf. equivalents; K.epsilon is now a constant 1e-7
        Shape will be  [input_shape[0], input_shape[1]]
        """
        attention_weight = attention / (
                tf.reduce_sum(attention, axis=-1, keepdims=True) + tf.constant(1e-7)
        )

        """
        Step 6: Expand the dim. of the attention
        Result is shape [input_shape[0], input_shape[1], 1]        
        attention-weight now holds a tensor where the first dim is over 
        the number of the elements of the batch, the second is how much 
        we care about each of the inputs for a given record, the third is void (since it has only 1 
        as dimensionality)   
         
        Original code: attention_weight = K.expand_dims(attention_weight)
        """
        attention_weight = tf.expand_dims(attention_weight, -1)

        """
        Step 7: Multiply input with attention weights. The two are both 3-dim tensors, one with shape 
        [input_shape[0], input_shape[1], input_shape[2]], the other with shape [input_shape[0], input_shape[1], 1].
        We do element-wise multiplication, and through broadcasting this works as expected.
        
        Original code: weighted_input = inputs * attention_weight
        """
        weighted_input = tf.multiply(inputs, attention_weight)

        """
        Step 8: Sum over the middle dim: Together with the previous scaling this gives the weighted sum
        we wanted. 
        
        Original code: outputs = K.sum(weighted_input, axis=1)         
        """
        outputs = tf.reduce_sum(weighted_input, axis=1, keepdims=False)

        return outputs

    # noinspection PyMethodMayBeStatic
    def compute_output_shape(self, child_input_shape):
        """
        Have to help Keras to understand the output shape.
        It will be whatever we get in after dropping the dim we attend over
        :param child_input_shape: Output-shape of the layer below
        :return: Output-shape: (batch-size, whatever is left over of dims when we attend/average over the first)
        """
        if len(child_input_shape) <= 2:
            raise ValueError(f"Trying to do ChineseAttention with input-shape {child_input_shape}. "
                             f"Don't know how to assess the output shape when input-shape is not at least 3D")

        output_shape = tuple([child_input_shape[0]]) + child_input_shape[2:]
        return output_shape


class Slice(tf.keras.layers.Layer):
    """
    Slice will take a slice of the input-layer and push that through.
    The news-encoder uses Lambda-layers for this, which is a nightmare.
    Slicing is done in the first data-dim (so, not batch-size-dim, but the following one.
    Whatever is not picked up by the slicing [:, begin : begin+size] is just dropped.
    """

    def __init__(self, begin: int = 0, size: int = 1, **kwargs):
        # Check params
        if begin < 0 or not isinstance(size, int):
            raise ValueError(f"Slice fails: 'begin' should be a non-neg integer, but was {begin}")
        if size <= 0 or not isinstance(size, int):
            raise ValueError(f"Slice fails: 'size' should be a positive integer, but was {size}")

        super(Slice, self).__init__(**kwargs)
        self.begin = begin
        self.size = size

    def get_config(self):
        """
        This is to fix so that the model can be saved: Need to be able to dump all internals
        """
        config = super(Slice, self).get_config().copy()
        config.update({
            'begin': self.begin,
            'size': self.size,
        })
        return config

    def call(self, inputs):
        if inputs.shape[1] < self.begin + self.size:
            raise ValueError(f"Slice fails: Shape is {inputs.shape}. \n"
                             f"Slice-params are begin: {self.begin}, size: {self.size}, so "
                             f"the intention is to slice [: {self.begin} : {self.begin + self.size}].")

        # I think I can do this using  return tf.slice(inputs, self.begin, self.size), but
        # the version below is more readable and feels better
        return inputs[:, self.begin: self.begin + self.size]

    def compute_output_shape(self, child_input_shape):
        """
        Have to help Keras to understand the output shape.
        It will be the shape of the part "cut out"
        :param child_input_shape: Output-shape of the layer "below"
        :return: Output-shape: (batch-size, self.size, whatever is left over of dims). Example:
            If input is a batch of tren elements, and we get  [10, 9, 8, 7, 6, 5] with size = 2,
            the return will be [10, 2, 8, 7, 6, 5]
        """
        output_shape = child_input_shape[0], self.size
        if len(child_input_shape) > 2:
            output_shape += child_input_shape[2:]
        return output_shape

    """
    def check_me(self, inputs):
        sliced = self.call(inputs=inputs).numpy()
        if isinstance(inputs, tf.Tensor):
            inputs = inputs.numpy()
        golden = inputs[:, self.begin:self.begin + self.size, ]
        if {golden.shape} == {sliced.shape}:
            print(f"shapes: {golden.shape} == {sliced.shape}, as desired!")
            print(f"Average error per entry: {np.mean(np.abs(sliced - golden)):.8f}")
        else:
            print(f"Error. Shapes are {golden.shape} and {sliced.shape}")
    """


class Slice2(tf.keras.layers.Layer):
    """
    Slice will take a slice of the input-layer and push that through.
    The news-encoder uses Lambda-layers for this, which is a nightmare.
    Slicing is done in the first data-dim (so, not batch-size-dim, but the following one.
    Whatever is not picked up by the slicing [:, begin : begin+size] is just dropped.
    """

    def __init__(self, begin: int = 0, size: int = 1, **kwargs):
        # Check params
        if begin < 0 or not isinstance(size, int):
            raise ValueError(f"Slice fails: 'begin' should be a non-neg integer, but was {begin}")
        if size <= 0 or not isinstance(size, int):
            raise ValueError(f"Slice fails: 'size' should be a positive integer, but was {size}")

        super(Slice2, self).__init__(**kwargs)
        self.begin = begin
        self.size = size

    def get_config(self):
        """
        This is to fix so that the model can be saved: Need to be able to dump all internals
        """
        config = super(Slice2, self).get_config().copy()
        config.update({
            'begin': self.begin,
            'size': self.size,
        })
        return config

    def call(self, inputs):
        if inputs.shape[2] < self.begin + self.size:
            raise ValueError(f"Slice fails: Shape is {inputs.shape}. \n"
                             f"Slice-params are begin: {self.begin}, size: {self.size}, so "
                             f"the intention is to slice [: {self.begin} : {self.begin + self.size}].")

        # I think I can do this using  return tf.slice(inputs, self.begin, self.size), but
        # the version below is more readable and feels better
        return inputs[:, :, self.begin: self.begin + self.size]

    def compute_output_shape(self, child_input_shape):
        """
        Have to help Keras to understand the output shape.
        It will be the shape of the part "cut out"
        :param child_input_shape: Output-shape of the layer "below"
        :return: Output-shape: (batch-size, self.size, whatever is left over of dims). Example:
            If input is a batch of tren elements, and we get  [10, 9, 8, 7, 6, 5] with size = 2,
            the return will be [10, 2, 8, 7, 6, 5]
        """
        output_shape = child_input_shape[0], child_input_shape[1], self.size
        if len(child_input_shape) > 2:
            output_shape += child_input_shape[2:]
        return output_shape

    """
    def check_me(self, inputs):
        sliced = self.call(inputs=inputs).numpy()
        if isinstance(inputs, tf.Tensor):
            inputs = inputs.numpy()
        golden = inputs[:, self.begin:self.begin + self.size, ]
        if {golden.shape} == {sliced.shape}:
            print(f"shapes: {golden.shape} == {sliced.shape}, as desired!")
            print(f"Average error per entry: {np.mean(np.abs(sliced - golden)):.8f}")
        else:
            print(f"Error. Shapes are {golden.shape} and {sliced.shape}")
    """

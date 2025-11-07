import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="MyLayers")
class VanillaRNN(tf.keras.Model):
    """
    Simple vanilla RNN implementation from scratch for text language modeling.
    Compatible with mystery dataset text modeling codebase.
    """

    def __init__(self, vocab_size, hidden_size, seq_length, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_seq_length = seq_length

        # TODO: Initialize embedding layer for token embeddings
        self.embedding = tf.keras.layers.Embedding(vocab_size, hidden_size)
        # TODO: Initialize RNN weight matrices using self.add_weight from the Keras API
        self.w_x = self.add_weight(shape=(hidden_size, hidden_size), initializer="glorot_uniform", 
                                             dtype=tf.float32, trainable=True, name="input weights")
        self.w_h = self.add_weight(shape=(hidden_size, hidden_size), initializer="glorot_uniform", 
                                              dtype=tf.float32, trainable=True, name="hidden weights")
        self.b_h = self.add_weight(shape=(hidden_size), initializer="zeros",
                                           dtype=tf.float32, trainable=True, name="bias vector")
        # TODO: Initialize output projection layer to project to vocabulary size
        self.output_projection = tf.keras.layers.Dense(self.vocab_size)

    def call(self, inputs, training=None):
        """
        Forward pass for text language modeling.

        Args:
            inputs: Input token indices of shape [batch_size, seq_length]
            training: Training mode flag

        Returns:
            Logits of shape [batch_size, seq_length, vocab_size]
        """
        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]

        # 1. First, we pass the input tokens through the embedding layer
        embedded = self.embedding(inputs)  # [batch_size, seq_length, hidden_size]

        # 2. Now, we initalize the RNN hidden state to zeros and we will update it step by step
        h = tf.zeros([batch_size, self.hidden_size])

        # 3. Now, iterate through the sequence length (one "token" at a time) and compute RNN hidden states
        outputs = []
        for t in range(seq_length):
            x_t = embedded[:, t, :]  # [batch_size, hidden_size]
            # TODO: Compute RNN hidden state update using the RNN equation(s)
            h = tf.nn.tanh(tf.matmul(x_t, self.w_x) + tf.matmul(h, self.w_h) + self.b_h)
            outputs.append(h)

        # 4. Finally, we stack the outputs and project to vocabulary
        stacked_outputs = tf.stack(outputs, axis=1)  # [batch_size, seq_length, hidden_size]
        logits = self.output_projection(stacked_outputs)  # [batch_size, seq_length, vocab_size]

        return logits

    def get_config(self):
        base_config = super().get_config()
        config = {k:getattr(self, k) for k in ["vocab_size", "hidden_size", "seq_length"]}
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

########################################################################################

@tf.keras.utils.register_keras_serializable(package="MyLayers")
class LSTM(tf.keras.Model):
    """
    LSTM implementation for comparison with vanilla RNN.
    """

    def __init__(self, vocab_size, hidden_size, seq_length, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_seq_length = seq_length

        # TODO: Initialize embedding layer for token embeddings
        self.embedding = tf.keras.layers.Embedding(vocab_size, hidden_size)
        # TODO: Initialize LSTM weight matrices and biases using self.add_weight
        self.w_f = self.add_weight(shape=(hidden_size, hidden_size), initializer="glorot_uniform", 
                                             dtype=tf.float32, trainable=True, name="forget gate input weights")
        self.u_f = self.add_weight(shape=(hidden_size, hidden_size), initializer="glorot_uniform", 
                                             dtype=tf.float32, trainable=True, name="forget gate hidden weights")
        self.b_f = self.add_weight(shape=(hidden_size), initializer="ones",
                                           dtype=tf.float32, trainable=True, name="forget gate bias vector")
        
        self.w_i = self.add_weight(shape=(hidden_size, hidden_size), initializer="glorot_uniform", 
                                             dtype=tf.float32, trainable=True, name="input gate input weights")
        self.u_i = self.add_weight(shape=(hidden_size, hidden_size), initializer="glorot_uniform", 
                                             dtype=tf.float32, trainable=True, name="input gate hidden weights")
        self.b_i = self.add_weight(shape=(hidden_size), initializer="ones",
                                           dtype=tf.float32, trainable=True, name="input gate bias vector")
        
        self.w_c = self.add_weight(shape=(hidden_size, hidden_size), initializer="glorot_uniform", 
                                             dtype=tf.float32, trainable=True, name="candidate cell state input weights")
        self.u_c = self.add_weight(shape=(hidden_size, hidden_size), initializer="glorot_uniform", 
                                             dtype=tf.float32, trainable=True, name="candidate cell state hidden weights")
        self.b_c = self.add_weight(shape=(hidden_size), initializer="zeros",
                                           dtype=tf.float32, trainable=True, name="candidate cell state bias vector")
        
        self.w_o = self.add_weight(shape=(hidden_size, hidden_size), initializer="glorot_uniform", 
                                             dtype=tf.float32, trainable=True, name="output gate input weights")
        self.u_o = self.add_weight(shape=(hidden_size, hidden_size), initializer="glorot_uniform", 
                                             dtype=tf.float32, trainable=True, name="output gate hidden weights")
        self.b_o = self.add_weight(shape=(hidden_size), initializer="zeros",
                                           dtype=tf.float32, trainable=True, name="output gate bias vector")
        # TODO: Initialize output layer to project to vocabulary size
        self.output_projection = tf.keras.layers.Dense(self.vocab_size)

    def call(self, inputs, training=None):
        """
        LSTM forward pass.

        Args:
            inputs: Input token indices [batch_size, seq_length]
            training: Training mode flag

        Returns:
            Logits [batch_size, seq_length, vocab_size]
        """
        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]

        # TODO: Embed input tokens
        embedded = self.embedding(inputs)  # [batch_size, seq_length, hidden_size]

        # TODO: Initialize hidden and cell states
        h = tf.zeros([batch_size, self.hidden_size]) # hidden state
        c = tf.zeros([batch_size, self.hidden_size]) # cell state

        # TODO: Process sequence step by step with LSTM equations
        outputs = []
        for t in range(seq_length):
            x_t = embedded[:, t, :]  # [batch_size, hidden_size]
            f_t = tf.nn.sigmoid(tf.linalg.matmul(x_t, self.w_f) + tf.linalg.matmul(h, self.u_f) + self.b_f) # forget gate
            i_t = tf.nn.sigmoid(tf.linalg.matmul(x_t, self.w_i) + tf.linalg.matmul(h, self.u_i) + self.b_i) # input gate
            cand_c = tf.nn.tanh(tf.linalg.matmul(x_t, self.w_c) + tf.linalg.matmul(h, self.u_c) + self.b_c) # candidate cell state
            c = f_t * c + i_t * cand_c # cell state update
            o_t = tf.nn.sigmoid(tf.linalg.matmul(x_t, self.w_o) + tf.linalg.matmul(h, self.u_o) + self.b_o) # output gate
            h = o_t * tf.nn.tanh(c)
            outputs.append(h)

        # TODO: Stack outputs and compute logits of each output step
        stacked_outputs = tf.stack(outputs, axis=1)  # [batch_size, seq_length, hidden_size]
        logits = self.output_projection(stacked_outputs)  # [batch_size, seq_length, vocab_size]

        return logits

    def get_config(self):
        base_config = super().get_config()
        config = {k:getattr(self, k) for k in ["vocab_size", "hidden_size", "seq_length"]}
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

def create_rnn_language_model(vocab_size, hidden_size, seq_length, model_type="vanilla"):
    """
    Create an RNN-based language model.

    Args:
        vocab_size: Size of vocabulary
        hidden_size: Hidden state dimension
        seq_length: Maximum sequence length
        model_type: Type of RNN ("vanilla", "lstm")

    Returns:
        Configured RNN language model
    """
    # 1. Create appropriate RNN layer based on model_type
    if model_type == "vanilla":
        model = VanillaRNN(vocab_size, hidden_size, seq_length)
    elif model_type == "lstm":
        model = LSTM(vocab_size, hidden_size, seq_length)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'vanilla' or 'lstm'")

    return model
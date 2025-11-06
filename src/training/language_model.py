import tensorflow as tf
import math
from typing import Optional, Tuple, Dict, Any



class TextSampler:
    """
    Advanced sampling methods for text generation.
    Students will implement top-k and top-p sampling methods.
    """

    @staticmethod
    def sample_top_k(logits: tf.Tensor, k: int, temperature: float = 1.0) -> tf.Tensor:
        """
        Sample from top-k most likely tokens.

        Args:
            logits: Raw model outputs [batch_size, vocab_size]
            k: Number of top tokens to consider
            temperature: Sampling temperature (higher = more random)

        Returns:
            Sampled token indices [batch_size, 1]
        """
        # TODO: Apply temperature scaling
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        scaled_logits = logits / temperature

        # TODO: Limit to top-k from the logits using tf.nn.top_k
        top_k = tf.nn.top_k(scaled_logits, k, sorted=True) # values, indices
        min_top_ks = top_k.values[:,-1] # list of minimum top k value for each input in batch

        # TODO: Filter logits to only keep top-k tokens, set others to -inf
        # filtered_logits = tf.zeros(shape=scaled_logits.shape, dtype=tf.float32)
        # for num_input_in_batch in range(scaled_logits.shape[0]):
        #     filtered_logits[num_input_in_batch, :] = tf.where(scaled_logits>=min_top_ks[num_input_in_batch], scaled_logits, neg_inf)

        # scaled_logits is of shape [batch_size, vocab_size]. For each batch, we need to set all logits less than the value of the relevant batch 
        # index in min_top_ks to neg_inf.
        update_indices_list = []
        for i in range(scaled_logits.shape[0]): # iterate over batch_size of scaled_logits
            for j in range(scaled_logits.shape[1]): # iterate over vocab_size of scaled_logits
                if scaled_logits[i,j] < min_top_ks[i]: # if the logit is less than the minimum top k value
                    update_indices_list.append(tf.constant([i,j], dtype=tf.int32)) # add the indices of the logit in scaled_logits to a list

        update_indices = tf.stack(update_indices_list) # stack the list into a tensor

        neg_inf = tf.cast(tf.fill(len(update_indices_list), -10000000), dtype=tf.float32)

        filtered_logits = tf.tensor_scatter_nd_update(scaled_logits, update_indices, neg_inf) # at all indices, set the logit to -inf

        # TODO: Sample from the filtered distribution
        return tf.random.categorical(filtered_logits, num_samples=1)

    @staticmethod
    def sample_top_p(logits: tf.Tensor, p: float, temperature: float = 1.0) -> tf.Tensor:
        """
        Sample from nucleus (top-p) of probability distribution.

        Args:
            logits: Raw model outputs [batch_size, vocab_size]
            p: Cumulative probability threshold (0.0 to 1.0)
            temperature: Sampling temperature

        Returns:
            Sampled token indices [batch_size, 1]
        """
        # TODO: Apply temperature scaling
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        scaled_logits = logits / temperature

        # TODO: Sort logits in descending order and prepare batch indices
        sorted_logits = tf.nn.top_k(scaled_logits, scaled_logits.shape[1], sorted=True) # values, indices
        # sorted_logits.values is of shape [batch_size, vocab_size]
        # sorted_logits.indices is also of shape [batch_size, vocab_size]

        # TODO: Compute cumulative probabilities from sorted logits
        cum_probs = tf.cumsum(sorted_logits.values, axis=1) # sum along axis vocab_size, produces shape [batch_size, vocab_size]

        # TODO: Create mask for tokens where cumulative probability <= p
        update_indices_list = []

        for i in range(cum_probs.shape[0]): # iterate over examples in batch
            passed_p = False
            for j in range(cum_probs.shape[1]): # iterate over cum probs in example
                if passed_p: # if the logit corresponding to this cum prob shouldn't be sampled anymore
                    # find the index of this logit in scaled_logits and add it to update_indices_list
                    indices = (i, sorted_logits.indices[i,j]) # find the indices of this logit in scaled_logits, [batch_size, vocab_size]
                    update_indices_list.append(indices)
                if j > p: # if the current cum prob is lower than the cum prob threshold
                    passed_p = True

        # TODO: Apply mask to filter sorted logits

        # TODO: Map filtered logits back to original indices and sample

        update_indices = tf.stack(update_indices_list) # stack the list into a tensor

        neg_inf = tf.cast(tf.fill(len(update_indices_list), -10000000), dtype=tf.float32)

        filtered_logits = tf.tensor_scatter_nd_update(scaled_logits, update_indices, neg_inf) # at all indices, set the logit to -inf

        return tf.random.categorical(filtered_logits, num_samples=1)

    @staticmethod
    def sample_categorical(logits: tf.Tensor, temperature: float = 1.0) -> tf.Tensor:
        """
        Basic categorical sampling (baseline implementation).

        Args:
            logits: Raw model outputs [batch_size, vocab_size]
            temperature: Sampling temperature

        Returns:
            Sampled token indices [batch_size, 1]
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        scaled_logits = logits / temperature
        return tf.random.categorical(scaled_logits, num_samples=1)

    @classmethod
    def sample(cls, logits: tf.Tensor, method: str = "temperature", **kwargs) -> tf.Tensor:
        """
        Unified sampling interface.

        Args:
            logits: Raw model outputs [batch_size, vocab_size]
            method: Sampling method ("temperature", "top_k", "top_p")
            **kwargs: Method-specific parameters

        Returns:
            Sampled token indices [batch_size, 1]
        """
        if method == "temperature":
            return cls.sample_categorical(logits, temperature=kwargs.get("temperature", 1.0))
        elif method == "top_k":
            return cls.sample_top_k(
                logits,
                k=kwargs.get("k", 50),
                temperature=kwargs.get("temperature", 1.0)
            )
        elif method == "top_p":
            return cls.sample_top_p(
                logits,
                p=kwargs.get("p", 0.9),
                temperature=kwargs.get("temperature", 1.0)
            )
        else:
            raise ValueError(f"Unknown sampling method: {method}")


class TextGenerator:
    """
    High-level text generation interface.
    """

    def __init__(self, model, tokenizer=None):
        """
        Initialize text generator.

        Args:
            model: Trained model
            tokenizer: Tokenizer for encoding/decoding text
        """
        self.model = model
        self.tokenizer = tokenizer
        self.sampler = TextSampler()

    def generate(self,
                 prompt: str = "",
                 max_length: int = 100,
                 method: str = "top_k",
                 temperature: float = 0.8,
                 top_k: int = 50,
                 top_p: float = 0.9,
                 stop_tokens: Optional[list] = None) -> str:
        """
        Generate text continuation from a prompt.

        Args:
            prompt: Starting text (empty string for unconditioned generation)
            max_length: Maximum tokens to generate
            method: Sampling method ("temperature", "top_k", "top_p")
            temperature: Sampling temperature
            top_k: Top-k parameter
            top_p: Top-p parameter
            stop_tokens: List of token IDs to stop generation

        Returns:
            Generated text string
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for text generation")
        
        # Encode prompt
        if prompt:
            prompt_tokens = self.tokenizer.encode(prompt)
        else:
            prompt_tokens = [1]  # Start token

        # We initalize the generated sequence with the prompt tokens and generate step by step
        generated = tf.constant([prompt_tokens], dtype=tf.int64)

        # Get max sequence length from model
        if hasattr(self.model, 'max_seq_length'):
            model_max_seq_length = self.model.max_seq_length
        else:
            # Otherwise, we can just infer from input shape
            model_max_seq_length = self.model.input_shape[1]

        # Generate tokens
        for _ in range(max_length):
            # Get current sequence
            current_seq = generated
            seq_len = tf.shape(current_seq)[1]
            
            if seq_len > model_max_seq_length:
                # Truncate to max length
                current_seq = current_seq[:, -model_max_seq_length:]
            elif seq_len < model_max_seq_length:
                # Pad to max length
                padding_length = model_max_seq_length - seq_len
                padding = tf.zeros([1, padding_length], dtype=tf.int64)
                # Pad on the left to preserve recent context
                current_seq = tf.concat([padding, current_seq], axis=-1)

            # Forward pass
            logits = self.model(current_seq, training=False)
            
            # Get logits for the last non padded position
            # For padded sequences, we need to get logits at position (original_seq_len - 1)
            if seq_len < model_max_seq_length:
                # Account for padding so we get logits at the last actual token position
                next_token_logits = logits[:, seq_len - 1, :]
            else:
                # No padding, use last position
                next_token_logits = logits[:, -1, :]

            # Sample next token using specified method
            sampling_kwargs = {
                "temperature": temperature,
                "k": top_k,
                "p": top_p
            }

            next_token = self.sampler.sample(next_token_logits, method=method, **sampling_kwargs)

            # Check for stop tokens
            if stop_tokens and int(next_token[0, 0]) in stop_tokens:
                break

            # Append to sequence
            generated = tf.concat([generated, next_token], axis=-1)

        # Decode to text
        generated_tokens = generated.numpy()[0].tolist()
        return self.tokenizer.decode(generated_tokens)
"""
train.py: Training utilities and loops for Transformer Language Model
TensorFlow implementation for mystery corpus training

Author: Eric Ewing
"""

import tensorflow as tf
import math
import os
import json
from typing import Tuple, Dict
import tqdm

# Import wandb for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def calculate_perplexity(loss: float) -> float:
    """Calculate perplexity from cross-entropy loss."""
    return math.exp(loss)

def train(model, train_dataset, test_dataset, epochs=5, learning_rate=1e-3,
          wandb_run=None, checkpoint_dir="checkpoints", continue_training=False, submission_tracker=None) -> Tuple[tf.keras.Model, Dict[str, list]]:
    """
    Complete training function for language models.

    Args:
        model: Language model to train
        train_dataset: Training dataset
        test_dataset: Test dataset
        epochs: Number of epochs
        learning_rate: Learning rate
        wandb_run: Wandb run for logging
        tokenizer: Tokenizer for text generation
        checkpoint_dir: Directory to save checkpoints
        continue_training: Whether to continue training from latest checkpoint
        submission_tracker: Submission tracker for logging epoch results

    Returns:
        model: Trained model
    """

    def get_input_target_seqs(sequence):
        input = sequence[:,:-1]
        target = sequence[:,1:]
        return input, target

    # Ensure checkpoint directory exists otherwise create it
    os.makedirs(checkpoint_dir, exist_ok=True) 
    # TODO: Initialize your optimizer and loss function
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # TODO: Set up TensorFlow checkpointing with Checkpoint and CheckpointManager
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=3)
    
    # Handle checkpoint restoration for continue training
    start_epoch = 1
    if continue_training:
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint)
            # Extract epoch number from checkpoint name
            try:
                start_epoch = int(latest_checkpoint.split('-')[-1])
                print(f"Resuming from epoch {start_epoch}")
            except:
                print("Could not determine start epoch, starting from 0")
        else:
            print("No checkpoint found, starting fresh")
    
    # This is to keep track of model's performance during training
    history = {'train_loss': [], 'val_loss': [], 'perplexity': []}

    # TODO: Train your model, keep track of metrics and log to wandb
    # NOTE: tqdm can be used to create progress bars for any iterable (epochs, batches, etc.)
    # You might find it useful to wrap your epoch loop with tqdm.tqdm(...) for visual feedback
    for current_epoch in tqdm.tqdm(range(start_epoch, start_epoch + epochs), desc="Training Progress", position=0):
        total_epochs = start_epoch + epochs - 1
        # TODO: Iterate over the training dataset and update model weights
        train_losses = []
        for batch in train_dataset:
            input, target = get_input_target_seqs(batch)

            with tf.GradientTape() as tape:
                logits = model(input)
                loss = tf.reduce_mean(loss_fn(y_true=tf.reshape(target, shape=(logits.shape[0] * logits.shape[1])), 
                                              y_pred=tf.reshape(logits, shape=(logits.shape[0] * logits.shape[1], logits.shape[2]))))
                # VERIFY THAT THE tf.reshape() ABOVE WORKS!

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_losses.append(loss)

        epoch_train_loss = sum(train_losses)/len(train_losses)

        # TODO: Iterate over the test dataset and compute validation loss
        # NOTE: Make sure to call reduce_mean on the loss
        test_losses = []
        for batch in test_dataset:
            input, target = get_input_target_seqs(batch)

            logits = model(input)
            loss = tf.reduce_mean(loss_fn(y_true=tf.reshape(target, shape=(logits.shape[0] * logits.shape[1])), 
                                              y_pred=tf.reshape(logits, shape=(logits.shape[0] * logits.shape[1], logits.shape[2]))))
                # VERIFY THAT THE tf.reshape() ABOVE WORKS!
            test_losses.append(loss)

        epoch_test_loss = sum(test_losses)/len(test_losses)

        # TODO: Calculate perplexity from validation loss
        # NOTE: Make sure to call reduce_mean on the loss
        perplexity = calculate_perplexity(epoch_test_loss)
        
        # TODO: Append metrics to history dictionary
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_test_loss)
        history['perplexity'].append(perplexity)

        # TODO: Log epoch metrics to the submission tracker (epoch, train_loss, val_loss, perplexity)
        if submission_tracker is not None:
            submission_tracker.log_epoch(current_epoch, epoch_train_loss, epoch_test_loss, perplexity)

        # TODO : Save model checkpoint periodically or if validation loss improves
        val_loss_increased = False
        if len(history['val_loss']) > 3:
            val_loss_increased = history['val_loss'][-2] - epoch_test_loss >= 0.3

        if current_epoch % 10 == 0 or val_loss_increased:
            checkpoint_manager.save()

        # Log metrics to wandb if available (recommended into batch loop for logging for better tracking)
        # NOTE: If using for batch, make sure to log epoch number, not batch number on some N interval
        if wandb_run:
            wandb_run.log({
                "epoch": current_epoch, # TODO: Current epoch number (one-index, so add 1
                "train_loss": epoch_train_loss,  # TODO: Calculate training loss
                "val_loss": epoch_test_loss,  # TODO: Calculate validation loss
                "perplexity": perplexity  # TODO: Calculate perplexity
            })
        

    return model, history
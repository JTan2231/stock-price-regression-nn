import os

import numpy as np
import pandas as pd
import subprocess as sp
import tensorflow as tf
import tensorflow.keras as keras

from transformer import Transformer, create_look_ahead_mask, create_padding_mask
from tqdm import tqdm
from random import randrange, sample

FEATURES = ["Open", "High", "Low", "Close", "Volume"]

BATCH_SIZE = 256
SEQUENCE_LENGTH = 50
WINDOW_SIZE = 100
MOMENTUM_WINDOW_SIZE = 25

EPOCHS = 20

VALIDATION_RATIO = 0.2

datapath = "data/stocks/"
all_stocks = [datapath+s for s in os.listdir(datapath)]

stock_array = []

for f in tqdm(all_stocks):
    df = pd.read_csv(f).drop('Date', axis=1)

    if df.shape[0] > WINDOW_SIZE:
        start = randrange(df.shape[0]-WINDOW_SIZE)
        stock_array.append(np.array(df)[start:start+WINDOW_SIZE])

stock_array = np.array(stock_array)[:,:,:-1]
stock_array[:,:,-1] /= 1e6

moving_averages = np.array([np.array([np.array(pd.Series(stock_array[x,:,i]).rolling(window=SEQUENCE_LENGTH).mean().fillna(0)) for x in range(stock_array.shape[0])]) for i in range(5)])
moving_averages = np.transpose(moving_averages, (1, 2, 0))
moving_stds = np.array([np.array([np.array(pd.Series(stock_array[x,:,i]).rolling(window=SEQUENCE_LENGTH).std().fillna(0)) for x in range(stock_array.shape[0])]) for i in range(5)])
moving_stds = np.transpose(moving_stds, (1, 2, 0))

moving_volatilities = moving_stds * np.sqrt(np.reshape(np.arange(moving_stds.shape[1])+1, (1, moving_stds.shape[1], 1)))

delayed_stocks = np.concatenate([np.zeros((stock_array.shape[0], MOMENTUM_WINDOW_SIZE, stock_array.shape[-1])),
                                 stock_array[:,:-MOMENTUM_WINDOW_SIZE]], axis=1)

momentums = stock_array - delayed_stocks
momentums[:,:MOMENTUM_WINDOW_SIZE] = np.zeros(())

rates_of_change = np.divide(momentums, delayed_stocks, where=delayed_stocks>0)
rates_of_change[np.abs(rates_of_change) < 1e-6] = np.zeros(())

cum_averages = [np.cumsum(stock_array[:,:,i], axis=1) / (np.arange(stock_array.shape[1])+1) for i in range(5)]
cum_averages = [np.expand_dims(x, axis=-1) for x in cum_averages]

stock_array = np.concatenate([stock_array,
                              moving_averages,
                              moving_stds,
                              moving_volatilities,
                              momentums,
                              rates_of_change]+cum_averages, axis=-1)

FEATURE_COUNT = stock_array.shape[-1]
STEPS_PER_EPOCH = stock_array.shape[0]

def generate_dataset(train_stock_array, validation_stock_array, train_steps, validation_steps):
    def train_generator():
        for i in range(train_steps):
            stock_index = randrange(train_stock_array.shape[0])
            start_index = randrange(train_stock_array.shape[1]-SEQUENCE_LENGTH)

            array = train_stock_array[stock_index,start_index:start_index+SEQUENCE_LENGTH]
            array[:,3] /= np.max(array[:,3])

            yield array, array[:,3:4]

    def validation_generator():
        for i in range(validation_steps):
            stock_index = randrange(validation_stock_array.shape[0])
            start_index = randrange(validation_stock_array.shape[1]-SEQUENCE_LENGTH)

            array = validation_stock_array[stock_index,start_index:start_index+SEQUENCE_LENGTH]
            array[:,3] /= np.max(array[:,3])

            yield array, array[:,3:4]

    train_dataset = tf.data.Dataset.from_generator(train_generator, output_types=(tf.float32, tf.float32))
    train_dataset = train_dataset.batch(BATCH_SIZE).repeat()

    validation_dataset = tf.data.Dataset.from_generator(validation_generator, output_types=(tf.float32, tf.float32))
    validation_dataset = validation_dataset.batch(BATCH_SIZE)

    return train_dataset, validation_dataset

VALIDATION_SIZE = int(stock_array.shape[0] * VALIDATION_RATIO)

validation_indices = sample(range(stock_array.shape[0]), VALIDATION_SIZE)
train_indices = [x for x in range(stock_array.shape[0]) if x not in validation_indices]

train_dataset, validation_dataset = generate_dataset(stock_array[train_indices], stock_array[validation_indices], STEPS_PER_EPOCH*BATCH_SIZE, VALIDATION_SIZE)

model = Transformer(2, 256, 8, 128, 1, 1, 10000, 10000)

model(tf.random.uniform((BATCH_SIZE, SEQUENCE_LENGTH, FEATURE_COUNT)),
      tf.random.uniform((BATCH_SIZE, SEQUENCE_LENGTH, 1)), False, None, None, None)

model.summary()

opt = keras.optimizers.Adam()
lf = keras.losses.MSE
#loss_function = lambda y, x: tf.nn.compute_average_loss(tf.math.sqrt(lf(y, x)))
#loss_function = lambda y, x: tf.nn.compute_average_loss(keras.losses.MSE(y, x))
def loss_function(target, logits):
    delta_squared = 1 / (BATCH_SIZE-1) * tf.math.reduce_sum(tf.math.square(target - tf.math.reduce_mean(target)))

    loss = 1 / (delta_squared*BATCH_SIZE) * tf.math.reduce_sum(tf.math.square(target - logits))

    return loss

@tf.function
def train_step(input_tensor, target):
    encoder_input = input_tensor
    decoder_input = tf.concat([tf.fill((input_tensor.shape[0], 1, 1), -1.), target[:,:-1,:]], axis=1)

    encoder_padding_mask = create_padding_mask(encoder_input[:,:,0])
    #decoder_padding_mask = create_padding_mask(decoder_input[:,0])

    look_ahead_mask = create_look_ahead_mask(SEQUENCE_LENGTH)

    with tf.GradientTape() as tape:
        logits = model(encoder_input, decoder_input, True, encoder_padding_mask,
                       look_ahead_mask, None)

        loss = loss_function(target, logits)

    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, logits

@tf.function
def eval_step(input_tensor, target):
    encoder_input = input_tensor
    decoder_input = tf.concat([tf.fill((input_tensor.shape[0], 1, 1), -1.), target[:,:-1,:]], axis=1)

    encoder_padding_mask = create_padding_mask(encoder_input[:,:,0])
    #decoder_padding_mask = create_padding_mask(decoder_input[:,0])

    look_ahead_mask = create_look_ahead_mask(SEQUENCE_LENGTH)

    logits = model(encoder_input, decoder_input, False, encoder_padding_mask,
                   look_ahead_mask, None)

    loss = loss_function(target, logits)

    return loss#, logits

steps_per_epoch = STEPS_PER_EPOCH
learning_rate_base = 1e-3
total_steps = steps_per_epoch * EPOCHS
warmup_learning_rate = 1e-5
warmup_steps = (EPOCHS // 10) * steps_per_epoch

#@tf.function
def cosine_decay_with_warmup(global_step,
                             hold_base_rate_steps=0):

    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                     'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + tf.cos(
        np.pi *
        (tf.cast(global_step, tf.float32) - warmup_steps - hold_base_rate_steps
        ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = tf.where(
          global_step > warmup_steps + hold_base_rate_steps,
          learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                         'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * tf.cast(global_step,
                                    tf.float32) + warmup_learning_rate
        learning_rate = tf.where(global_step < warmup_steps, warmup_rate,
                               learning_rate)
    return tf.where(global_step > total_steps, 0.0, learning_rate,
                    name='learning_rate')

class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __call__(self, step):
        return cosine_decay_with_warmup(step)

schedule = LRSchedule()

#tf.debugging.enable_check_numerics()

loss_met = keras.metrics.Mean()
it = iter(train_dataset)
for e in range(EPOCHS):
    print(f"Epoch {e}/{EPOCHS}")

    bar = tqdm(range(STEPS_PER_EPOCH))
    for i in bar:
        inp, tar = next(it)

        loss, logits = train_step(inp, tar)
        loss_met(loss)

        bar.set_description(f"loss: {loss_met.result():0.5f}")

        opt.learning_rate.assign(cosine_decay_with_warmup(e*STEPS_PER_EPOCH+i))

    val_loss_met = keras.metrics.Mean()
    for vi, vt in validation_dataset:
        val_loss_met(eval_step(vi, vt))

    print(f"Train loss: {loss_met.result()}, validation loss: {val_loss_met.result()}")

    loss_met.reset_states()

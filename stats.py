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
WINDOW_SIZE = 100
SEQUENCE_LENGTH = WINDOW_SIZE // 2
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

label_array = stock_array[:,:,3:4]
stock_array = np.concatenate([stock_array[:,:,:3], stock_array[:,:,4:]], axis=-1)

moving_averages = np.array([np.array([np.array(pd.Series(stock_array[x,:,i]).rolling(window=SEQUENCE_LENGTH).mean().fillna(0)) for x in range(stock_array.shape[0])]) for i in range(4)])
moving_averages = np.transpose(moving_averages, (1, 2, 0))
moving_stds = np.array([np.array([np.array(pd.Series(stock_array[x,:,i]).rolling(window=SEQUENCE_LENGTH).std().fillna(0)) for x in range(stock_array.shape[0])]) for i in range(4)])
moving_stds = np.transpose(moving_stds, (1, 2, 0))

moving_volatilities = moving_stds * np.sqrt(np.reshape(np.arange(moving_stds.shape[1])+1, (1, moving_stds.shape[1], 1)))

delayed_stocks = np.concatenate([np.zeros((stock_array.shape[0], MOMENTUM_WINDOW_SIZE, stock_array.shape[-1])),
                                 stock_array[:,:-MOMENTUM_WINDOW_SIZE]], axis=1)

momentums = stock_array - delayed_stocks
momentums[:,:MOMENTUM_WINDOW_SIZE] = np.zeros(())

rates_of_change = np.divide(momentums, delayed_stocks, where=delayed_stocks>0)
rates_of_change[np.abs(rates_of_change) < 1e-6] = np.zeros(())

cum_averages = [np.cumsum(stock_array[:,:,i], axis=1) / (np.arange(stock_array.shape[1])+1) for i in range(4)]
cum_averages = [np.expand_dims(x, axis=-1) for x in cum_averages]

stock_array = np.concatenate([stock_array,
                              moving_averages,
                              moving_stds,
                              moving_volatilities,
                              momentums,
                              rates_of_change]+cum_averages, axis=-1)

FEATURE_COUNT = stock_array.shape[-1]+1
print(f"Feature count: {FEATURE_COUNT}")
STEPS_PER_EPOCH = stock_array.shape[0]

def generate_dataset(train_stock_arrays, validation_stock_arrays, train_steps, validation_steps):
    def train_generator():
        for i in range(train_stock_arrays[0].shape[0]):
            stock_index = i
            #start_index = randrange(train_stock_arrays[0].shape[1]-SEQUENCE_LENGTH)

            array = train_stock_arrays[0][stock_index,:SEQUENCE_LENGTH]#,start_index:start_index+SEQUENCE_LENGTH]
            #array[:,3] /= np.max(array[:,3])

            all_close = train_stock_arrays[1][stock_index]
            label_array = all_close[SEQUENCE_LENGTH:]#,start_index:start_index+SEQUENCE_LENGTH]

            array = np.concatenate([array, all_close[:SEQUENCE_LENGTH]], axis=-1)

            yield array, label_array

    def validation_generator():
        for i in range(validation_stock_arrays[0].shape[0]):
            stock_index = i
            #start_index = randrange(validation_stock_arrays[0].shape[1]-SEQUENCE_LENGTH)

            array = validation_stock_arrays[0][stock_index,:SEQUENCE_LENGTH]#,start_index:start_index+SEQUENCE_LENGTH]
            #array[:,3] /= np.max(array[:,3])

            all_close = validation_stock_arrays[1][stock_index]
            label_array = all_close[SEQUENCE_LENGTH:]#,start_index:start_index+SEQUENCE_LENGTH]

            array = np.concatenate([array, all_close[:SEQUENCE_LENGTH]], axis=-1)

            yield array, label_array

    train_dataset = tf.data.Dataset.from_generator(train_generator, output_types=(tf.float32, tf.float32))
    train_dataset = train_dataset.batch(BATCH_SIZE).repeat()

    validation_dataset = tf.data.Dataset.from_generator(validation_generator, output_types=(tf.float32, tf.float32))
    validation_dataset = validation_dataset.batch(BATCH_SIZE)

    return train_dataset, validation_dataset

VALIDATION_SIZE = int(stock_array.shape[0] * VALIDATION_RATIO)

validation_indices = sample(range(stock_array.shape[0]), VALIDATION_SIZE)
train_indices = [x for x in range(stock_array.shape[0]) if x not in validation_indices]

train_dataset, validation_dataset = generate_dataset((stock_array[train_indices], label_array[train_indices]),
                                                     (stock_array[validation_indices], label_array[validation_indices]),
                                                     STEPS_PER_EPOCH*BATCH_SIZE, VALIDATION_SIZE)

model = Transformer(2, 512, 8, 256, 10000, 10000)

model(tf.random.uniform((BATCH_SIZE, SEQUENCE_LENGTH, FEATURE_COUNT)),
      tf.random.uniform((BATCH_SIZE, SEQUENCE_LENGTH, 1)), False, None, None, None)

model.summary()

opt = keras.optimizers.Adam()
lf = keras.losses.MSE
#loss_function = lambda y, x: tf.nn.compute_average_loss(tf.math.sqrt(lf(y, x)))
#loss_function = lambda y, x: tf.nn.compute_average_loss(keras.losses.MSE(y, x))

def nmse(target, logits):
    delta_squared = tf.math.reduce_mean(tf.math.square(target - tf.math.reduce_mean(target)))
    loss = (1 / delta_squared) * tf.math.reduce_mean(tf.math.square(target - logits))

    return loss

def loss_function(target, logits):
    smoothing = 0.1

    regression_pred = logits[:,:,0:1]
    regression_loss = nmse(target, regression_pred)
 
    target_off = tf.concat([tf.zeros((target.shape[0], 1, target.shape[-1])), target[:,:-1]], axis=1)
    regression_off = tf.concat([tf.zeros((logits.shape[0], 1, regression_pred.shape[-1])), regression_pred[:,:-1]], axis=1)

    target_diff = (target - target_off)[:,1:]
    regression_diff = (regression_pred - regression_off)[:,1:]

    diffs_loss = nmse(target_diff, regression_diff)

    #up_pred = logits[:,1:,1:2]
    #down_pred = logits[:,1:,2:]
    up_label = tf.cast(target_diff > 0, tf.float32)
    down_label = tf.cast(target_diff < 0, tf.float32)

    #up_loss = keras.losses.binary_crossentropy(up_label, tf.clip_by_value(regression_diff, 0+smoothing, 1-smoothing), label_smoothing=smoothing)
    #up_loss = tf.math.reduce_sum(up_loss) / tf.math.reduce_sum(up_label)
    #down_loss = keras.losses.binary_crossentropy(down_label, tf.clip_by_value(regression_diff, smoothing-1, smoothing-0), label_smoothing=smoothing)
    #down_loss = tf.math.reduce_sum(down_loss) / tf.math.reduce_sum(down_label)

    loss = regression_loss + diffs_loss# + up_loss + down_loss

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

    return loss, logits

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

# TODO: comment on what these are
class NMSE(keras.metrics.Metric):
    def __init__(self):
        super().__init__()

        self.met = keras.metrics.Mean()

    def update_state(self, target, logits, sample_weight=None):
        output = nmse(target, logits)

        self.met.update_state(output)

    def reset_states(self):
        self.met.reset_states()

    def result(self):
        return self.met.result()

class DirectionalSymmetry(keras.metrics.Metric):
    def __init__(self):
        super().__init__()

        self.met = keras.metrics.Mean()

    def update_state(self, target, logits, sample_weight=None):
        target_off = tf.concat([tf.zeros((target.shape[0], 1, target.shape[-1])), target[:,:-1]], axis=1)
        logits_off = tf.concat([tf.zeros((logits.shape[0], 1, logits.shape[-1])), logits[:,:-1]], axis=1)

        target_diff = (target - target_off)[:,1:]
        logits_diff = (logits - logits_off)[:,1:]

        output = tf.cast((target_diff * logits_diff) >= 0, tf.float32)
        output = tf.math.reduce_mean(output)

        self.met.update_state(output)

    def reset_states(self):
        self.met.reset_states()

    def result(self):
        return self.met.result()

class CorrectUp(keras.metrics.Metric):
    def __init__(self):
        super().__init__()

        self.met = keras.metrics.Mean()

    def update_state(self, target, logits, sample_weight=None):
        target_off = tf.concat([tf.zeros((target.shape[0], 1, target.shape[-1])), target[:,:-1]], axis=1)
        logits_off = tf.concat([tf.zeros((logits.shape[0], 1, logits.shape[-1])), logits[:,:-1]], axis=1)

        target_diff = (target - target_off)[:,1:]
        logits_diff = (logits - logits_off)[:,1:]

        output = tf.math.logical_and(logits_diff > 0, (target_diff * logits_diff) >= 0)
        output = tf.math.reduce_mean(tf.cast(output, tf.float32))

        self.met.update_state(output)

    def reset_states(self):
        self.met.reset_states()

    def result(self):
        return self.met.result()

class CorrectDown(keras.metrics.Metric):
    def __init__(self):
        super().__init__()

        self.met = keras.metrics.Mean()

    def update_state(self, target, logits, sample_weight=None):
        target_off = tf.concat([tf.zeros((target.shape[0], 1, target.shape[-1])), target[:,:-1]], axis=1)
        logits_off = tf.concat([tf.zeros((logits.shape[0], 1, logits.shape[-1])), logits[:,:-1]], axis=1)

        target_diff = (target - target_off)[:,1:]
        logits_diff = (logits - logits_off)[:,1:]

        output = tf.math.logical_and(logits_diff < 0, (target_diff * logits_diff) >= 0)
        output = tf.math.reduce_mean(tf.cast(output, tf.float32))

        self.met.update_state(output)

    def reset_states(self):
        self.met.reset_states()

    def result(self):
        return self.met.result()

#tf.debugging.enable_check_numerics()

metrics = { "mae": keras.metrics.MeanAbsoluteError(),
            "nmse": NMSE(),
            "directional_symmetry": DirectionalSymmetry(),
            "correct_up": CorrectUp(),
            "correct_down": CorrectDown() }

loss_met = keras.metrics.Mean()
it = iter(train_dataset)
for e in range(EPOCHS):
    print(f"Epoch {e}/{EPOCHS}")

    bar = tqdm(range(STEPS_PER_EPOCH))
    for i in bar:
        inp, tar = next(it)

        loss, logits = train_step(inp, tar)
        regression_pred = logits[:,:,:1]
        loss_met(loss)

        results = dict()
        for key, metric in metrics.items():
            metric.update_state(tar, regression_pred)
            results[key] = metric.result().numpy()

        bar.set_description(f"loss: {loss_met.result():.4f}, "+", ".join([key+": "+f"{result:.4f}" for key, result in results.items()]))

        opt.learning_rate.assign(cosine_decay_with_warmup(e*STEPS_PER_EPOCH+i))

    val_metrics = { "mae": keras.metrics.MeanAbsoluteError(),
                    "nmse": NMSE(),
                    "directional_symmetry": DirectionalSymmetry(),
                    "correct_up": CorrectUp(),
                    "correct_down": CorrectDown() }

    val_loss_met = keras.metrics.Mean()
    for vi, vt in validation_dataset:
        loss, logits = eval_step(vi, vt)
        regression_pred = logits[:,:,:1]

        val_loss_met(loss)
        for key, metric in val_metrics.items():
            metric.update_state(vt, regression_pred)

    print(f"train_loss: {loss_met.result():.4f}, val_loss: {val_loss_met.result():.4f}, ", end='')
    print(", ".join([key+": "+f"{metric.result():.4f}" for key, metric in val_metrics.items()]))

    for _, metric in metrics.items():
        metric.reset_states()

    loss_met.reset_states()

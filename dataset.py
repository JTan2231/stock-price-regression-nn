import os

import numpy as np
import pandas as pd
import tensorflow as tf

from params import params
from random import randrange, sample
from tqdm import tqdm

def prepare_data():
    datapath = "data/stocks/"
    all_stocks = [datapath+s for s in os.listdir(datapath)]

    stock_array = []

    for f in tqdm(all_stocks):
        df = pd.read_csv(f).drop('Date', axis=1)

        if df.shape[0] > params.WINDOW_SIZE:
            start = randrange(df.shape[0]-params.WINDOW_SIZE)
            stock_array.append(np.array(df)[start:start+params.WINDOW_SIZE])

    stock_array = np.array(stock_array)[:,:,:-1]
    stock_array[:,:,-1] /= 1e6

    label_array = stock_array[:,:,3:4]
    stock_array = np.concatenate([stock_array[:,:,:3], stock_array[:,:,4:]], axis=-1)

    moving_averages = np.array(
        [np.array([
            np.array(pd.Series(
                stock_array[x,:,i]).rolling(
                    window=params.SEQUENCE_LENGTH).mean().fillna(0)
                ) for x in range(stock_array.shape[0])]
            ) for i in range(4)]
        )
    moving_averages = np.transpose(moving_averages, (1, 2, 0))
    moving_stds = np.array(
        [np.array([
            np.array(pd.Series(
                stock_array[x,:,i]).rolling(
                    window=params.SEQUENCE_LENGTH).std().fillna(0)
                ) for x in range(stock_array.shape[0])]
            ) for i in range(4)]
        )
    moving_stds = np.transpose(moving_stds, (1, 2, 0))

    moving_volatilities = moving_stds * np.sqrt(np.reshape(np.arange(moving_stds.shape[1])+1, (1, moving_stds.shape[1], 1)))

    delayed_stocks = np.concatenate([np.zeros((stock_array.shape[0], params.MOMENTUM_WINDOW_SIZE, stock_array.shape[-1])),
                                     stock_array[:,:-params.MOMENTUM_WINDOW_SIZE]], axis=1)

    momentums = stock_array - delayed_stocks
    momentums[:,:params.MOMENTUM_WINDOW_SIZE] = np.zeros(())

    rates_of_change = np.divide(momentums, delayed_stocks, where=delayed_stocks>0)
    rates_of_change[np.abs(rates_of_change) < 1e-6] = np.zeros(())

    cum_averages = [np.cumsum(stock_array[:,:,i], axis=1) / (np.arange(stock_array.shape[1])+1) for i in range(4)]
    cum_averages = [np.expand_dims(x, axis=-1) for x in cum_averages]

    stock_array = np.concatenate([stock_array]+cum_averages+[moving_averages,
                                                             moving_stds,
                                                             moving_volatilities,
                                                             momentums,
                                                             rates_of_change], axis=-1)

    params.FEATURE_COUNT = stock_array.shape[-1]+1
    print(f"Feature count: {params.FEATURE_COUNT}")

    def generate_dataset(train_stock_arrays, validation_stock_arrays, train_steps, validation_steps):
        def train_generator():
            for i in range(train_stock_arrays[0].shape[0]):
                stock_index = i
                #start_index = randrange(train_stock_arrays[0].shape[1]-SEQUENCE_LENGTH)

                array = train_stock_arrays[0][stock_index,:params.SEQUENCE_LENGTH]#,start_index:start_index+SEQUENCE_LENGTH]
                #array[:,3] /= np.max(array[:,3])

                all_close = train_stock_arrays[1][stock_index]
                label_array = all_close[params.SEQUENCE_LENGTH:]#,start_index:start_index+SEQUENCE_LENGTH]

                array = np.concatenate([array, all_close[:params.SEQUENCE_LENGTH]], axis=-1)

                yield array, label_array

        def validation_generator():
            for i in range(validation_stock_arrays[0].shape[0]):
                stock_index = i
                #start_index = randrange(validation_stock_arrays[0].shape[1]-SEQUENCE_LENGTH)

                array = validation_stock_arrays[0][stock_index,:params.SEQUENCE_LENGTH]#,start_index:start_index+SEQUENCE_LENGTH]
                #array[:,3] /= np.max(array[:,3])

                all_close = validation_stock_arrays[1][stock_index]
                label_array = all_close[params.SEQUENCE_LENGTH:]#,start_index:start_index+SEQUENCE_LENGTH]

                array = np.concatenate([array, all_close[:params.SEQUENCE_LENGTH]], axis=-1)

                yield array, label_array

        train_dataset = tf.data.Dataset.from_generator(train_generator, output_types=(tf.float32, tf.float32))
        train_dataset = train_dataset.batch(params.BATCH_SIZE).repeat()

        validation_dataset = tf.data.Dataset.from_generator(validation_generator, output_types=(tf.float32, tf.float32))
        validation_dataset = validation_dataset.batch(params.BATCH_SIZE)

        return train_dataset, validation_dataset

    params.VALIDATION_SIZE = int(stock_array.shape[0] * params.VALIDATION_RATIO)
    print(f"VALIDATION_SIZE: {params.VALIDATION_SIZE}")

    validation_indices = sample(range(stock_array.shape[0]), params.VALIDATION_SIZE)
    train_indices = [x for x in range(stock_array.shape[0]) if x not in validation_indices]
    params.STEPS_PER_EPOCH = len(train_indices)
    print(f"STEPS_PER_EPOCH: {params.STEPS_PER_EPOCH}")

    train_dataset, validation_dataset = generate_dataset((stock_array[train_indices], label_array[train_indices]),
                                                         (stock_array[validation_indices], label_array[validation_indices]),
                                                         params.STEPS_PER_EPOCH*params.BATCH_SIZE, params.VALIDATION_SIZE)

    return train_dataset, validation_dataset

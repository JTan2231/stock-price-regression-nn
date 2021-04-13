import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from params import params

def nmse(target, logits):
    delta_squared = tf.math.reduce_mean(tf.math.square(target - tf.math.reduce_mean(target)))
    loss = (1 / delta_squared) * tf.math.reduce_mean(tf.math.square(target - logits))

    return loss

steps_per_epoch = params.STEPS_PER_EPOCH
learning_rate_base = 1e-3
total_steps = steps_per_epoch * params.EPOCHS
warmup_learning_rate = 1e-5
warmup_steps = (params.EPOCHS // 10) * steps_per_epoch

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

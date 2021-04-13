import tensorflow as tf
import tensorflow.keras as keras

from transformer import Transformer, create_padding_mask, create_look_ahead_mask
from params import params
from train_utils import nmse

optimizers = {
    'adam': keras.optimizers.Adam(),
    'sgd': keras.optimizers.SGD()
}

class MarketPredictionModel:
    def __init__(self):
        super().__init__()

        self.nn = Transformer(params.LAYERS, params.D_MODEL, params.HEADS, params.DFF)

        self.SEQUENCE_LENGTH = params.SEQUENCE_LENGTH

        encoder_input_shape = [params.BATCH_SIZE, params.SEQUENCE_LENGTH, params.FEATURE_COUNT]
        decoder_input_shape = [params.BATCH_SIZE, params.SEQUENCE_LENGTH, 1]

        self.nn(tf.random.uniform(encoder_input_shape),
                tf.random.uniform(decoder_input_shape), False, None, None, None)

        #self.nn.build([tf.TensorShape(encoder_input_shape),
        #               tf.TensorShape(decoder_input_shape),
        #               tf.TensorShape(None),
        #               tf.TensorShape(None),
        #               tf.TensorShape(None),
        #               tf.TensorShape(None)])

        self.nn.summary()

        self.opt = optimizers[params.OPT]
        self.regression_loss = nmse

    def loss_function(self, target, direction_pred, regression_pred):
        smoothing = 0.1

        #regression_pred = logits[:,:,0:1]
        regression_loss = self.regression_loss(target, regression_pred)
     
        target_off = tf.concat([tf.zeros((target.shape[0], 1, target.shape[-1])), target[:,:-1]], axis=1)
        regression_off = tf.concat([tf.zeros((regression_pred.shape[0], 1, regression_pred.shape[-1])), regression_pred[:,:-1]], axis=1)

        target_diff = (target - target_off)[:,1:]
        regression_diff = (regression_pred - regression_off)[:,1:]

        diffs_loss = self.regression_loss(target_diff, regression_diff)

        up_pred = direction_pred[:,1:,:1]
        down_pred = direction_pred[:,1:,1:]

        up_label = tf.cast(target_diff > 0, tf.float32)
        down_label = tf.cast(target_diff < 0, tf.float32)

        up_loss = keras.losses.binary_crossentropy(up_label, up_pred)
        up_loss = tf.math.reduce_sum(up_loss) / tf.math.reduce_sum(up_label)
        down_loss = keras.losses.binary_crossentropy(down_label, down_pred)
        down_loss = tf.math.reduce_sum(down_loss) / tf.math.reduce_sum(down_label)

        loss = regression_loss + diffs_loss + up_loss + down_loss

        return loss

    #@tf.function
    def train_step(self, input_tensor, target):
        encoder_input = input_tensor
        decoder_input = tf.concat([tf.fill((input_tensor.shape[0], 1, 1), -1.), target[:,:-1,:]], axis=1)

        encoder_padding_mask = create_padding_mask(encoder_input[:,:,0])
        decoder_padding_mask = create_padding_mask(decoder_input[:,0])

        look_ahead_mask = create_look_ahead_mask(self.SEQUENCE_LENGTH)

        with tf.GradientTape() as tape:
            direction_pred, regression_pred = self.nn(encoder_input, decoder_input, True, encoder_padding_mask,
                                                      look_ahead_mask, decoder_padding_mask)

            loss = self.loss_function(target, direction_pred, regression_pred)

        gradients = tape.gradient(loss, self.nn.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.nn.trainable_variables))

        logits = tf.concat([regression_pred, direction_pred], axis=-1)

        return loss, logits

    #@tf.function
    def eval_step(self, input_tensor, target):
        encoder_input = input_tensor
        decoder_input = tf.concat([tf.fill((input_tensor.shape[0], 1, 1), -1.), target[:,:-1,:]], axis=1)

        encoder_padding_mask = create_padding_mask(encoder_input[:,:,0])
        decoder_padding_mask = create_padding_mask(decoder_input[:,0])

        look_ahead_mask = create_look_ahead_mask(self.SEQUENCE_LENGTH)

        direction_pred, regression_pred = self.nn(encoder_input, decoder_input, False, encoder_padding_mask,
                                                  look_ahead_mask, None)

        loss = self.loss_function(target, direction_pred, regression_pred)

        logits = tf.concat([regression_pred, direction_pred], axis=-1)

        return loss, logits

import train_utils as utils
import tensorflow as tf
import tensorflow.keras as keras

from tqdm import tqdm
from params import params

def train(model, datasets):
    train_dataset, validation_dataset = datasets

    metrics = { "mae": keras.metrics.MeanAbsoluteError(),
                "nmse": utils.NMSE(),
                "directional_symmetry": utils.DirectionalSymmetry(),
                "correct_up": utils.CorrectUp(),
                "correct_down": utils.CorrectDown() }

    loss_met = keras.metrics.Mean()
    it = iter(train_dataset)
    for e in range(params.EPOCHS):
        print("==============================================================")
        print(f"Epoch {e}/{params.EPOCHS}")

        bar = tqdm(range(params.STEPS_PER_EPOCH))
        for i in bar:
            inp, tar = next(it)

            loss, logits = model.train_step(inp, tar)
            regression_pred = logits[:,:,:1]
            loss_met(loss)

            results = dict()
            for key, metric in metrics.items():
                metric.update_state(tar, regression_pred)
                results[key] = metric.result().numpy()

            bar.set_description(f"loss: {loss_met.result():.4f}, "+", ".join([key+": "+f"{result:.4f}" for key, result in results.items()]))

            model.opt.learning_rate.assign(utils.cosine_decay_with_warmup(e*params.STEPS_PER_EPOCH+i))

        val_metrics = { "mae": keras.metrics.MeanAbsoluteError(),
                        "nmse": utils.NMSE(),
                        "directional_symmetry": utils.DirectionalSymmetry(),
                        "correct_up": utils.CorrectUp(),
                        "correct_down": utils.CorrectDown() }

        val_loss_met = keras.metrics.Mean()
        for vi, vt in validation_dataset:
            loss, logits = model.eval_step(vi, vt)
            regression_pred = logits[:,:,:1]

            val_loss_met(loss)
            for key, metric in val_metrics.items():
                metric.update_state(vt, regression_pred)

        print(f"train_loss: {loss_met.result():.4f}, val_loss: {val_loss_met.result():.4f}, ", end='')
        print(", ".join([key+": "+f"{metric.result():.4f}" for key, metric in val_metrics.items()]))

        for _, metric in metrics.items():
            metric.reset_states()

        loss_met.reset_states()

        model.nn.save_weights("weights.h5")

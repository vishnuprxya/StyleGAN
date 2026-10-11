"""Adversarial training loop (5 epochs) with style mixing + label smoothing."""
import os, csv, random
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from tensorflow import keras
from data.dataset import load_fashion_mnist
from models import StyleGenerator, build_discriminator

SEED, LATENT, EPOCHS, BATCH = 42, 128, 5, 128
MIX_PROB, REAL_LABEL = 0.9, 0.9
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

G, D = StyleGenerator(LATENT), build_discriminator()
G(tf.random.normal([2, LATENT]))  # build weights
g_opt = keras.optimizers.Adam(2e-4, beta_1=0.5, beta_2=0.99)
d_opt = keras.optimizers.Adam(2e-4, beta_1=0.5, beta_2=0.99)
bce = keras.losses.BinaryCrossentropy(from_logits=True)


@tf.function
def train_step(real, crossover):  # crossover: python int, 0 = no mixing
    B = tf.shape(real)[0]
    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
        w = G.mapping(tf.random.normal([B, LATENT]))
        w2 = G.mapping(tf.random.normal([B, LATENT])) if crossover > 0 else None
        fake = G.synthesize(w, w2, crossover if crossover > 0 else None)
        r_log, f_log = D(real, training=True), D(fake, training=True)
        d_loss = bce(tf.fill([B, 1], REAL_LABEL), r_log) + bce(tf.zeros([B, 1]), f_log)
        g_loss = bce(tf.ones([B, 1]), f_log)
    d_opt.apply_gradients(zip(d_tape.gradient(d_loss, D.trainable_variables), D.trainable_variables))
    g_opt.apply_gradients(zip(g_tape.gradient(g_loss, G.trainable_variables), G.trainable_variables))
    return d_loss, g_loss


if __name__ == "__main__":
    ds = load_fashion_mnist(BATCH, SEED)
    os.makedirs("checkpoints", exist_ok=True); os.makedirs("outputs", exist_ok=True)
    rows = []
    for ep in range(1, EPOCHS + 1):
        d_m, g_m = keras.metrics.Mean(), keras.metrics.Mean()
        for real in ds:
            cross = random.randint(1, 3) if random.random() < MIX_PROB else 0
            d_l, g_l = train_step(real, cross)
            d_m.update_state(d_l); g_m.update_state(g_l)
        rows.append((ep, float(d_m.result()), float(g_m.result())))
        print(f"Epoch {ep}/{EPOCHS}  D_loss={rows[-1][1]:.4f}  G_loss={rows[-1][2]:.4f}")
    G.save_weights("checkpoints/netG.weights.h5")
    with open("outputs/training_log.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch", "d_loss", "g_loss"]); w.writerows(rows)
    print("Saved checkpoints/netG.weights.h5 and outputs/training_log.csv")

"""Style-mixing grid and latent LERP plots (max 4 images per plot)."""
import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _show(img):
    return np.clip((img.numpy() + 1) / 2, 0, 1)


def style_mixing_grid(G, out_path, seed=7, crossover=2):
    tf.random.set_seed(seed)
    zA, zB = tf.random.normal([1, G.latent_dim]), tf.random.normal([1, G.latent_dim])
    wA, wB = G.mapping(zA), G.mapping(zB)
    imgs = [
        ("Source Style A (w1)", G.synthesize(wA)[0]),
        ("Source Style B (w2)", G.synthesize(wB)[0]),
        ("Mixed: Style A + Style B", G.synthesize(wA, wB, crossover)[0]),
        ("Mixed: Style B + Style A", G.synthesize(wB, wA, crossover)[0]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6, 6.5))
    for ax, (t, im) in zip(axes.ravel(), imgs):
        ax.imshow(_show(im)); ax.set_title(t, fontweight="bold"); ax.axis("off")
    fig.suptitle("StyleGAN Style Mixing (4 Sample Grid)", fontweight="bold")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def latent_interpolation(G, out_path, seed=11):
    tf.random.set_seed(seed)
    wA = G.mapping(tf.random.normal([1, G.latent_dim]))
    wB = G.mapping(tf.random.normal([1, G.latent_dim]))
    alphas = [0.0, 0.33, 0.67, 1.0]
    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    for ax, a in zip(axes, alphas):
        ax.imshow(_show(G.synthesize((1 - a) * wA + a * wB)[0]))
        ax.set_title(f"\u03b1 = {a:.2f}"); ax.axis("off")
    fig.suptitle("Latent Space LERP Interpolation (4 Steps)", fontweight="bold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)

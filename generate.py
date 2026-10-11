"""Inference: style-mixing grid + latent interpolation figures."""
import argparse, os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from models import StyleGenerator
from utils.visualize import style_mixing_grid, latent_interpolation

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", default="checkpoints/netG.weights.h5")
p.add_argument("--out_dir", default="outputs")
a = p.parse_args()

G = StyleGenerator(128)
G(tf.random.normal([2, 128]))      # build before loading
G.load_weights(a.checkpoint)
os.makedirs(a.out_dir, exist_ok=True)
style_mixing_grid(G, os.path.join(a.out_dir, "style_mixing.png"))
latent_interpolation(G, os.path.join(a.out_dir, "latent_interpolation.png"))
print("Saved figures to", a.out_dir)

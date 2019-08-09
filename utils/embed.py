from .parameters import Parameters
from . import dataset
import torch as t
from ..model.vae import VAE
import numpy as np


def load_model(checkpoint_file):
    checkpoint = t.load(checkpoint_file, map_location=lambda storage, loc: storage)
    parameters = Parameters(checkpoint["batch_loader"]["vocab_size"], embed_size=32)
    vae = VAE(parameters.vocab_size, parameters.embed_size, parameters.latent_size,
              parameters.decoder_rnn_size, parameters.decoder_rnn_num_layers, 209)
    vae.load_state_dict(checkpoint['model_state_dict'])

    vocab = (checkpoint["batch_loader"]["vocab_size"],
             checkpoint["batch_loader"]["idx_to_char"],
             checkpoint["batch_loader"]["char_to_idx"])

    return vae, vocab


def embed(sentence, checkpoint_file):
    vae, vocab = load_model(checkpoint_file)
    ds = dataset.DatasetSkeleton()
    ds.set_vocab(*vocab)
    torchified_tuple = ds.torchify_example(sentence[:210])
    encoder_input, _, _ = ds.collate_fn([torchified_tuple])

    vae.eval()
    encoder_input = vae.embed(encoder_input)
    context = vae.encoder(encoder_input)
    context = context.detach().numpy()
    return context

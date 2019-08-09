import argparse
import numpy as np
import os
import torch as t
from torch.optim import Adam
import torch.nn.functional as F
from .utils.batchloader import BatchLoader
from .utils.dataset import DatasetFromHdf5
from .utils.parameters import Parameters
from .model.vae import VAE
from torch.autograd import Variable
import torch.utils.data as data
import logging

logger = logging.getLogger(__name__)


def calculate_loss(input, decoder_input, target, vae, dropout, vocab_size):
    target = target.view(-1)

    logits, aux_logits, kld = vae(dropout, input, decoder_input)

    logits = logits.view(-1, vocab_size)
    cross_entropy = F.cross_entropy(logits, target, size_average=False)

    aux_logits = aux_logits.view(-1, vocab_size)
    aux_cross_entropy = F.cross_entropy(aux_logits, target, size_average=False)

    return cross_entropy, aux_cross_entropy, kld


def train(filename, num_iterations=35000, n_epochs=20, batch_size=300, use_cuda=True,
          learning_rate=0.0005, learning_rate_scale=100, dropout=0, aux=0.2,
          use_trained=None, kld_weight=4, embed_size=32, max_len=210, vocab=None, save_model_dir=None, save_log_dir="../logs"):
    wanted_keys = ['split', 'data_path', 'go_token', 'pad_token', 'stop_token', 'vocab_size',
                   'idx_to_char', 'char_to_idx', 'max_seq_len', 'data_len']
    save_string = "-".join([str(x) for x in [num_iterations, batch_size, learning_rate, learning_rate_scale, dropout, aux, kld_weight, embed_size, max_len]])

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                        filename=os.path.join(save_log_dir, '{}.log'.format(save_string)))

    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')

    console.setFormatter(formatter)
    if len(logger.handlers) < 1:
        logging.getLogger(__name__).addHandler(console)

    if vocab:
        batch_loader = BatchLoader(filename, max_seq_len=max_len-1, vocab=vocab)
    else:
        batch_loader = BatchLoader(filename, max_seq_len=max_len - 1)
    parameters = Parameters(batch_loader.vocab_size, embed_size)

    vae = VAE(parameters.vocab_size, parameters.embed_size, parameters.latent_size,
              parameters.decoder_rnn_size, parameters.decoder_rnn_num_layers, max_len-1)
    if use_cuda:
        vae = vae.cuda()
    optimizer = Adam(vae.parameters(), learning_rate)

    if use_trained:
        # vae.load_state_dict(t.load('trained_VAE'))
        checkpoint = t.load(use_trained)
        vae.load_state_dict(checkpoint['model_state_dict'])
        if use_cuda:
            device = t.device("cuda")
            vae.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        iteration = checkpoint['iteration']
        loss = checkpoint['loss']
        start_epoch = checkpoint['epoch'] + 1
        train_results = checkpoint['train_results']
        valid_results = checkpoint['valid_results']

        del batch_loader, parameters
        vocab = (checkpoint["batch_loader"]["vocab_size"],
                 checkpoint["batch_loader"]["idx_to_char"],
                 checkpoint["batch_loader"]["char_to_idx"])
        batch_loader = BatchLoader(filename, max_seq_len=max_len, vocab=vocab)
        parameters = Parameters(batch_loader.vocab_size, embed_size=embed_size)
    else:
        iteration = -1
        start_epoch = 0
        train_results = []
        valid_results = []

    for epoch in range(start_epoch, start_epoch+n_epochs):
        scheduler = t.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iterations, learning_rate / learning_rate_scale)
        for iteration in range(iteration+1, iteration + num_iterations):
            if kld_weight:
                kld_w = min(iteration / (num_iterations * kld_weight), 1.0)
            else:
                kld_w = 0

            '''Train step'''
            vae.train()
            input, decoder_input, target = batch_loader.next_batch(batch_size, 'train', use_cuda)

            target = target.view(-1)

            logits, aux_logits, kld = vae(dropout, input, decoder_input)

            logits = logits.view(-1, batch_loader.vocab_size)
            cross_entropy = F.cross_entropy(logits, target, size_average=False)

            aux_logits = aux_logits.view(-1, batch_loader.vocab_size)
            aux_cross_entropy = F.cross_entropy(aux_logits, target, size_average=False)

            loss = cross_entropy + aux * aux_cross_entropy + kld_w * kld

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            '''Validation'''
            vae.eval()
            input, decoder_input, target = batch_loader.next_batch(batch_size, 'valid', use_cuda)
            target = target.view(-1)

            logits, aux_logits, valid_kld = vae(dropout, input, decoder_input)

            logits = logits.view(-1, batch_loader.vocab_size)
            valid_cross_entropy = F.cross_entropy(logits, target, size_average=False)

            aux_logits = aux_logits.view(-1, batch_loader.vocab_size)
            valid_aux_cross_entropy = F.cross_entropy(aux_logits, target, size_average=False)

            loss = valid_cross_entropy + aux * valid_aux_cross_entropy + valid_kld

            if iteration % 50 == 0:
                logger.info('\n')
                logger.info('|--------------------------------------|')
                logger.info(iteration)
                logger.info('|--------ce------aux-ce-----kld--------|')
                logger.info('|----------------train-----------------|')
                train_res = (iteration, cross_entropy.data.cpu().numpy() / (max_len * batch_size),
                             aux_cross_entropy.data.cpu().numpy() / (max_len * batch_size),
                             kld.data.cpu().numpy())
                train_results.append(train_res)
                logger.info("%s %s %s %s", *train_res)
                logger.info('|----------------valid-----------------|')
                valid_res = (iteration, valid_cross_entropy.data.cpu().numpy() / (max_len * batch_size),
                             valid_aux_cross_entropy.data.cpu().numpy() / (max_len * batch_size),
                             valid_kld.data.cpu().numpy())
                valid_results.append(valid_res)
                logger.info("%s %s %s %s", *valid_res)
                logger.info('|--------------------------------------|')
                inputv, _, _ = batch_loader.next_batch(2, 'valid', use_cuda)
                logger.info("Input size: {}".format(inputv.size()))
                mu, logvar = vae.inference(inputv)
                mu = mu[0]
                logvar = logvar[0]
                std = t.exp(0.5 * logvar)

                z = Variable(t.randn([1, parameters.latent_size]))
                if use_cuda:
                    z = z.cuda()
                z = z * std + mu
                logger.info(''.join([batch_loader.idx_to_char[idx] for idx in inputv.data.cpu().numpy()[0]]))
                logger.info(">" + vae.sample(batch_loader, use_cuda, z))
                logger.info('|--------------------------------------|')
        if save_model_dir:
            t.save({
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss,
                'iteration': iteration,
                'train_results': train_results,
                'valid_results': valid_results,
                'parameters': parameters.__dict__,
                'batch_loader': {k: batch_loader.__dict__[k] for k in wanted_keys if k in batch_loader.__dict__},
                'epoch': epoch,
            }, os.path.join(save_model_dir, f"vae{epoch}_{save_string}.tar"))

def train_ds(filename, num_iterations=35000, n_epochs=20, batch_size=300, use_cuda=True,
          learning_rate=0.0005, learning_rate_scale=100, dropout=0, aux=0.2,
          use_trained=None, kld_weight=4, embed_size=32, max_len=210, vocab=None, save_model_dir=None, save_log_dir="../logs"):
    wanted_keys = ['split', 'data_path', 'go_token', 'pad_token', 'stop_token', 'vocab_size',
                   'idx_to_char', 'char_to_idx', 'max_seq_len', 'data_len']
    save_string = "-".join([str(x) for x in [num_iterations, batch_size, learning_rate, learning_rate_scale, dropout, aux, kld_weight, embed_size, max_len]])

    device = t.device("cuda:0" if use_cuda else "cpu")

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                        filename=os.path.join(save_log_dir, '{}.log'.format(save_string)))

    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    console.setFormatter(formatter)
    if len(logger.handlers) < 1:
        logging.getLogger(__name__).addHandler(console)

    ds = DatasetFromHdf5(filename, "train")
    ds_valid = DatasetFromHdf5(filename, "valid")
    if vocab:
        ds.set_vocab(*vocab)
        ds_valid.set_vocab(*vocab)
    else:
        vocab = ds.build_vocab()
        ds_valid.set_vocab(*vocab)
    parameters = Parameters(ds.vocab_size, embed_size)

    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 1,
              "collate_fn": ds.collate_fn}

    training_generator = data.DataLoader(ds, **params)
    params["num_workers"] = 0
    validation_generator = data.DataLoader(ds_valid, **params)

    vae = VAE(parameters.vocab_size, parameters.embed_size, parameters.latent_size,
              parameters.decoder_rnn_size, parameters.decoder_rnn_num_layers, max_len-1)
    if use_cuda:
        vae = vae.cuda()
    optimizer = Adam(vae.parameters(), learning_rate)

    if use_trained:
        # vae.load_state_dict(t.load('trained_VAE'))
        checkpoint = t.load(use_trained)
        vae.load_state_dict(checkpoint['model_state_dict'])
        if use_cuda:
            device = t.device("cuda")
            vae.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        iteration = checkpoint['iteration']
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        train_results = checkpoint['train_results']
        valid_results = checkpoint['valid_results']

        del parameters
        vocab = (checkpoint["batch_loader"]["vocab_size"],
                 checkpoint["batch_loader"]["idx_to_char"],
                 checkpoint["batch_loader"]["char_to_idx"])
        ds.set_vocab(*vocab)
        ds_valid.set_vocab(*vocab)
        parameters = Parameters(ds.vocab_size, embed_size=embed_size)
    else:
        iteration = 0
        start_epoch = 0
        train_results = []
        valid_results = []

    for epoch in range(start_epoch, start_epoch+n_epochs):
        scheduler = t.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iterations, learning_rate / learning_rate_scale)
        valid_iter = iter(validation_generator)
        for input, decoder_input, target in training_generator:
            input, decoder_input, target = input.to(device), decoder_input.to(device), target.to(device)
            if kld_weight:
                kld_w = min(iteration / (num_iterations * kld_weight), 1.0)
            else:
                kld_w = 0

            '''Train step'''
            vae.train()
            #input, decoder_input, target = batch_loader.next_batch(batch_size, 'train', use_cuda)

            target = target.view(-1)

            logits, aux_logits, kld = vae(dropout, input, decoder_input)

            logits = logits.view(-1, ds.vocab_size)
            cross_entropy = F.cross_entropy(logits, target, size_average=False)

            aux_logits = aux_logits.view(-1, ds.vocab_size)
            aux_cross_entropy = F.cross_entropy(aux_logits, target, size_average=False)

            loss = cross_entropy + aux * aux_cross_entropy + kld_w * kld

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            '''Validation'''
            vae.eval()
            try:
                input_v, decoder_input_v, target_v = next(valid_iter)
            except StopIteration:
                valid_iter = iter(validation_generator)
                input_v, decoder_input_v, target_v = next(valid_iter)

            input_v, decoder_input_v, target_v = input_v.to(device), decoder_input_v.to(device), target_v.to(device)
            target_v = target_v.view(-1)

            logits, aux_logits, valid_kld = vae(dropout, input_v, decoder_input_v)

            logits = logits.view(-1, ds.vocab_size)
            valid_cross_entropy = F.cross_entropy(logits, target_v, size_average=False)

            aux_logits = aux_logits.view(-1, ds.vocab_size)
            valid_aux_cross_entropy = F.cross_entropy(aux_logits, target_v, size_average=False)

            loss = valid_cross_entropy + aux * valid_aux_cross_entropy + valid_kld

            if iteration % 50 == 0:
                logger.info('\n')
                logger.info('|--------------------------------------|')
                logger.info(iteration)
                logger.info('|--------ce------aux-ce-----kld--------|')
                logger.info('|----------------train-----------------|')
                train_res = (iteration, cross_entropy.data.cpu().numpy() / (max_len * batch_size),
                             aux_cross_entropy.data.cpu().numpy() / (max_len * batch_size),
                             kld.data.cpu().numpy())
                train_results.append(train_res)
                logger.info("%s %s %s %s", *train_res)
                logger.info('|----------------valid-----------------|')
                valid_res = (iteration, valid_cross_entropy.data.cpu().numpy() / (max_len * batch_size),
                             valid_aux_cross_entropy.data.cpu().numpy() / (max_len * batch_size),
                             valid_kld.data.cpu().numpy())
                valid_results.append(valid_res)
                logger.info("%s %s %s %s", *valid_res)
                logger.info('|--------------------------------------|')
                #inputv, _, _ = batch_loader.next_batch(2, 'valid', use_cuda)
                logger.info("Input size: {}".format(input_v.size()))
                mu, logvar = vae.inference(input_v)
                mu = mu[0]
                logvar = logvar[0]
                std = t.exp(0.5 * logvar)

                z = Variable(t.randn([1, parameters.latent_size]))
                if use_cuda:
                    z = z.cuda()
                z = z * std + mu
                logger.info(''.join([ds.idx_to_char[idx] for idx in input_v.data.cpu().numpy()[0]]))
                logger.info(">" + vae.sample(ds, use_cuda, z))
                logger.info('|--------------------------------------|')

            iteration += 1
        if save_model_dir:
            t.save({
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss,
                'iteration': iteration,
                'train_results': train_results,
                'valid_results': valid_results,
                'epoch': epoch,
                'parameters': parameters.__dict__,
                'batch_loader': {k: ds.__dict__[k] for k in wanted_keys if k in ds.__dict__},
            }, os.path.join(save_model_dir, f"vae{epoch}_{save_string}.tar"))

# if __name__ == "__main__":
#
#     parser = argparse.ArgumentParser(description='VAE')
#     parser.add_argument('--num-iterations', type=int, default=200000, metavar='NI',
#                         help='num iterations (default: 200000)')
#     parser.add_argument('--batch-size', type=int, default=30, metavar='BS',
#                         help='batch size (default: 30)')
#     parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA',
#                         help='use cuda (default: False)')
#     parser.add_argument('--learning-rate', type=float, default=0.0005, metavar='LR',
#                         help='learning rate (default: 0.0005)')
#     parser.add_argument('--dropout', type=float, default=0.12, metavar='DR',
#                         help='dropout (default: 0.12)')
#     parser.add_argument('--aux', type=float, default=0.4, metavar='DR',
#                         help='aux loss coef (default: 0.4)')
#     parser.add_argument('--use-trained', type=bool, default=False, metavar='UT',
#                         help='load pretrained model (default: False)')
#
#     args = parser.parse_args()
#
#     batch_loader = BatchLoader()
#     parameters = Parameters(batch_loader.vocab_size)
#
#     vae = VAE(parameters.vocab_size, parameters.embed_size, parameters.latent_size,
#               parameters.decoder_rnn_size, parameters.decoder_rnn_num_layers)
#     if args.use_trained:
#         vae.load_state_dict(t.load('trained_VAE'))
#     if args.use_cuda:
#         vae = vae.cuda()
#
#     optimizer = Adam(vae.parameters(), args.learning_rate)
#
#     for iteration in range(args.num_iterations):
#
#         '''Train step'''
#         input, decoder_input, target = batch_loader.next_batch(args.batch_size, 'train', args.use_cuda)
#         target = target.view(-1)
#
#         logits, aux_logits, kld = vae(args.dropout, input, decoder_input)
#
#         logits = logits.view(-1, batch_loader.vocab_size)
#         cross_entropy = F.cross_entropy(logits, target, size_average=False)
#
#         aux_logits = aux_logits.view(-1, batch_loader.vocab_size)
#         aux_cross_entropy = F.cross_entropy(aux_logits, target, size_average=False)
#
#         loss = cross_entropy + args.aux * aux_cross_entropy + kld
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         '''Validation'''
#         input, decoder_input, target = batch_loader.next_batch(args.batch_size, 'valid', args.use_cuda)
#         target = target.view(-1)
#
#         logits, aux_logits, valid_kld = vae(args.dropout, input, decoder_input)
#
#         logits = logits.view(-1, batch_loader.vocab_size)
#         valid_cross_entropy = F.cross_entropy(logits, target, size_average=False)
#
#         aux_logits = aux_logits.view(-1, batch_loader.vocab_size)
#         valid_aux_cross_entropy = F.cross_entropy(aux_logits, target, size_average=False)
#
#         loss = valid_cross_entropy + args.aux * valid_aux_cross_entropy + kld
#
#         if iteration % 50 == 0:
#             print('\n')
#             print('|--------------------------------------|')
#             print(iteration)
#             print('|--------ce------aux-ce-----kld--------|')
#             print('|----------------train-----------------|')
#             print(cross_entropy.data.cpu().numpy()[0]/(210 * args.batch_size),
#                   aux_cross_entropy.data.cpu().numpy()[0]/(210 * args.batch_size),
#                   kld.data.cpu().numpy()[0])
#             print('|----------------valid-----------------|')
#             print(valid_cross_entropy.data.cpu().numpy()[0]/(210 * args.batch_size),
#                   valid_aux_cross_entropy.data.cpu().numpy()[0]/(210 * args.batch_size),
#                   valid_kld.data.cpu().numpy()[0])
#             print('|--------------------------------------|')
#             input, _, _ = batch_loader.next_batch(2, 'valid', args.use_cuda)
#             mu, logvar = vae.inference(input[0].unsqueeze(1))
#             std = t.exp(0.5 * logvar)
#
#             z = Variable(t.randn([1, parameters.latent_size]))
#             if args.use_cuda:
#                 z = z.cuda()
#             z = z * std + mu
#             print(''.join([batch_loader.idx_to_char[idx] for idx in input.data.cpu().numpy()[0]]))
#             print(vae.sample(batch_loader, args.use_cuda, z))
#             print('|--------------------------------------|')

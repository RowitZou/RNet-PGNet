# -*- coding: utf8 -*-

import argparse

################################################################################
# ArgParse and Helper Functions #
################################################################################


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--trainset', type=str, default=None, help='Training set')
    parser.add_argument('--devset', type=str, default=None, help='Dev set')
    parser.add_argument('--testset', type=str, default=None, help='Test set')
    parser.add_argument('--dir', type=str, default=None, help='Set the name of the models directory for this session.')
    parser.add_argument('--pretrained', type=str, default=None, help='Specify pretrained models directory.')
    parser.add_argument('--random_seed', type=int, default=123, help='Random seed')
    parser.add_argument('--n_history', type=int, default=0)
    parser.add_argument('--min_freq', type=int, default=1)

    group = parser.add_argument_group('model_spec')
    group.add_argument('--max_word_length', type=int, default=8, help='Set maximum word length.')
    group.add_argument('--embed_file', type=str, default=None)
    group.add_argument('--embed_size', type=int, default=None)
    group.add_argument('--embed_type', type=str, default='glove', choices=['glove', 'word2vec', 'fasttext'])
    group.add_argument('--hidden_size', type=int, default=100, help='Set hidden size.')
    group.add_argument('--sent_rnn_layers', type=int, default=3, help='Set sentence RNN encoder layers.')
    group.add_argument('--sum_loss', type=str2bool, default=False, help="Set the type of loss.")
    group.add_argument('--fix_embeddings', type=str2bool, default=True, help='Whether to fix embeddings.')
    group.add_argument('--dropout_rnn', type=float, default=0.2, help='Set RNN dropout in reader.')
    group.add_argument('--dropout_emb', type=float, default=0.3, help='Set dropout for all feedforward layers.')
    group.add_argument('--use_dot_attention', type=str2bool, default=True, help='Whether to use dot self matching.')
    group.add_argument('--use_multi_gpu', type=str2bool, default=False, help='Whether to use multiple gpus.')
    # Optimizer
    group = parser.add_argument_group('training_spec')
    group.add_argument('--optimizer', type=str, default='adamax', help='Set optimizer.')
    group.add_argument('--learning_rate', type=float, default=0.1, help='Set learning rate for SGD.')
    group.add_argument('--grad_clipping', type=float, default=10.0, help='Whether to use grad clipping.')
    group.add_argument('--weight_decay', type=float, default=0.0, help='Set weight decay.')
    group.add_argument('--momentum', type=float, default=0.0, help='Set momentum.')
    group.add_argument('--batch_size', type=int, default=32, help='Set batch size.')
    group.add_argument('--max_epochs', type=int, default=20, help='Set number of total epochs.')
    group.add_argument('--verbose', type=int, default=100, help='Print every X batches.')
    group.add_argument('--shuffle', type=str2bool, default=True,
                       help='Whether to shuffle the examples during training.')
    group.add_argument('--max_answer_len', type=int, default=15, help='Set max answer length for decoding.')
    group.add_argument('--predict_train', type=str2bool, default=True, help='Whether to predict on training set.')
    group.add_argument('--out_predictions', type=str2bool, default=True, help='Whether to output predictions.')
    group.add_argument('--save_params', type=str2bool, default=True, help='Whether to save params.')

    args = parser.parse_args()
    # args denotes the dict of parameters
    return vars(args)

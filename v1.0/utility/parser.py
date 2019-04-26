#!/usr/local/bin/bash
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run GAT.")
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Input data path.')
    parser.add_argument('--data_path', nargs='?', default='D:\PycharmProjects\\recommender-project\\v1.0\Data\\',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Input data path.')

    parser.add_argument('--dataset', nargs='?', default='debug\\',
                        help='Choose a dataset.')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, 1: Pretrain with updating FISM variables, 2:Pretrain with fixed FISM variables.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epoch.')
    parser.add_argument('--weight_size', type=int, default=8,
                        help='batch size.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='weight size.')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]',
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--sigma', type=float, default=0.5,
                        help='Index of coefficient of embedding vector')
    parser.add_argument('--omega', type=float, default=0.5,
                        help='Index of coefficient of embedding vector')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Index of coefficient of sum of exp(A)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--lr_decay', type=float, default=0.98,
                        help='Learning rate.')

    parser.add_argument('--loss_type', nargs='?', default='bpr_loss',
                        help='Specify a loss type (bpr_loss, square_loss or log_loss).')
    parser.add_argument('--model_type', nargs='?', default='fm',
                        help='Specify a loss type (pure_mf or gat_mf).')
    parser.add_argument('--kge_type', nargs='?', default='None',
                        help='Specify a loss type (None, TransE-l2 or TransE-log).')

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    parser.add_argument('--drop_ratio', type=float, default=0.8,
                        help='Learning rate decay factor.')
    parser.add_argument('--batch_norm', type=int, default=1,
                        help='Whether to perform batch normaization (0 or 1)')

    parser.add_argument('--relation_type', nargs='?', default='pure_uni',
                        help='Specify a loss type (None, TransE-l2 or TransE-log).')
    parser.add_argument('--layer_size', nargs='?', default='[64]',
                        help='Output sizes of every layer')
    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',
                        help='Output sizes of every layer')
    parser.add_argument('--keep_prob', nargs='?', default='[0.8]',
                        help='Keep probability (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1]',
                        help='Keep probability (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: No pretrain, 1: Pretrain with updating FISM variables, 2:Pretrain with fixed FISM variables.')
    parser.add_argument('--adj_type', nargs='?', default='si',
                        help='Specify a loss type (plain, norm, or mean).')
    parser.add_argument('--alg_type', nargs='?', default='org',
                        help='Specify a loss type (org, norm, or mean).')


    parser.add_argument('--pre_model', nargs='?', default='bprmf',
                        help='Specify a loss type (pure_mf or gat_mf).')
    parser.add_argument('--pre_lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--pre_regs', nargs='?', default='[1e-5,1e-5,1e-2]',
                        help='Regularization for user and item embeddings.')

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify a loss type (org, norm, or mean).')

    parser.add_argument('--mf_in', type=int, default=1,
                        help='0: No pretrain, 1: Pretrain with updating FISM variables, 2:Pretrain with fixed FISM variables.')
    parser.add_argument('--report', type=int, default=0,
                        help='0: No pretrain, 1: Pretrain with updating FISM variables, 2:Pretrain with fixed FISM variables.')
    return parser.parse_args()
#!/usr/local/bin/bash
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run FM.")
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Input data path.')
    parser.add_argument('--data_path', nargs='?', default='Data\\',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='D:\\PycharmProjects\\recommender-project\\v1.0\\',
                        help='Project path.')
    parser.add_argument('--dataset', nargs='?', default='',
                        help='Choose a dataset.')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, 1: Pretrain with updating FISM variables, 2:Pretrain with fixed FISM variables.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epoch.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='weight size.')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]',
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--lr_decay', type=float, default=0.98,
                        help='Learning rate.')
    parser.add_argument('--loss_type', nargs='?', default='log_loss',
                        help='Specify a loss type (bpr_loss, square_loss or log_loss).')
    parser.add_argument('--model_type', nargs='?', default='fm',
                        help='Specify a loss type (pure_mf or gat_mf).')
    parser.add_argument('--kge_type', nargs='?', default='None',
                        help='Specify a loss type (None, TransE-l2 or TransE-log).')
    parser.add_argument('--drop_ratio', type=float, default=0.8,
                        help='Learning rate decay factor.')
    parser.add_argument('--batch_norm', type=int, default=1,
                        help='Whether to perform batch normaization (0 or 1)')
    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: No pretrain, 1: Pretrain with updating FISM variables, 2:Pretrain with fixed FISM variables.')
    parser.add_argument('--pre_lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--pre_regs', nargs='?', default='[1e-5,1e-5,1e-2]',
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify a loss type (org, norm, or mean).')
    parser.add_argument('--report', type=int, default=0,
                        help='0: No pretrain, 1: Pretrain with updating FISM variables, 2:Pretrain with fixed FISM variables.')
    return parser.parse_args()
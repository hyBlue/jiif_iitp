import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='jiif')
    parser.add_argument('--model', type=str, default='JIIF')
    parser.add_argument('--loss', type=str, default='L1')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='NYU')
    parser.add_argument('--data_root', type=str, default='./data/nyu_labeled/')
    parser.add_argument('--train_batch', type=int, default=1)
    parser.add_argument('--test_batch', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epoch', default=100, type=int, help='max epoch')
    parser.add_argument('--eval_interval',  default=10,
                        type=int, help='eval interval')
    parser.add_argument('--checkpoint',  default='scratch',
                        type=str, help='checkpoint to use')
    parser.add_argument('--scale',  default=4, type=int, help='scale')
    parser.add_argument('--interpolation',  default='bicubic',
                        type=str, help='interpolation method to generate lr depth')
    parser.add_argument('--lr',  default=0.0001,
                        type=float, help='learning rate')
    parser.add_argument('--lr_step',  default=40, type=float,
                        help='learning rate decay step')
    parser.add_argument('--lr_gamma',  default=0.2,
                        type=float, help='learning rate decay gamma')
    parser.add_argument('--input_size',  default=None,
                        type=int, help='crop size for hr image')
    parser.add_argument('--sample_q',  default=30720,
                        type=int, help='sampled pixels per hr depth')
    parser.add_argument('--noisy',  action='store_true',
                        help='add noise to train dataset')
    parser.add_argument('--test',  action='store_true', help='test mode')
    parser.add_argument('--report_per_image',
                        action='store_true', help='report RMSE of each image')
    parser.add_argument('--save',  action='store_true', help='save results')
    parser.add_argument('--batched_eval',  action='store_true',
                        help='batched evaluation to avoid OOM for large image resolution')

    
    ## QUANTIZATION
    parser.add_argument('--quantize',  action='store_true',
                        help='save results')
    parser.add_argument("--mode", default='avgbit',
                        choices=['avgbit', 'bops'], help='average bit mode')

    parser.add_argument("--a_scale", default=1, type=float)
    parser.add_argument("--w_scale", default=1, type=float)
    parser.add_argument("--bops_scale", default=3,
                        type=float, help='using teacher')

    parser.add_argument("--target", default=8, type=float,
                        help='target bitops or avgbit')

    parser.add_argument(
        "--ckpt_path", help="checkpoint directory", default='./checkpoint')

    parser.add_argument("--warmuplen", default=3, type=int,
                        help='scheduler warm up epoch')
    parser.add_argument("--ft_epoch", default=3, type=int, help='tuning epoch')

    parser.add_argument("--ts", action='store_true', help='using teacher')

    parser.add_argument("--ts_model", type=str,
                        default='deeplabv3plus_mobilenet')
    parser.add_argument("--exp", type=str, default='')
    parser.add_argument("--optim", type=str, default='adam')
    parser.add_argument("--last_fp", action='store_true')
    parser.add_argument("--resume", action='store_true')

    ## PRUNING
    parser.add_argument("--pruning", action='store_true')     
    parser.add_argument("--prune_ratio_method", type=str, choices=['global', 'uniform', 'adaptive', 'manual'], default='uniform')   # Pruning ratio 적용 방식 지정
    parser.add_argument("--importance_metric", type=str, choices=['l1', 'l2', 'entropy'], default='l1')                             # Importance metric 지정
    parser.add_argument("--prune_ratio", type=float, default=0.7)                                                                   # Sparsity 지정 (= 1 - density)
    parser.add_argument("--applyFirstLastLayer", action='store_true')     
    parser.add_argument('--pruner', choices=['str', 'wgt'], default='wgt')                                   # Pruner 유형 선택 : Channel(str), Weight(wgt)
    parser.add_argument("--iterative", action='store_true')                                                                         # Iterative Pruning (if false: One-shot)
    parser.add_argument("--iter_num", type=int, default=5)
    parser.add_argument("--KD", action='store_true')                        # Iterative KD 활용 여부

    args = parser.parse_args()
    return args

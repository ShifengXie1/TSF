import argparse
import copy
from datetime import datetime
import json
import os
import shutil
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
import numpy as np


def build_run_tag(ii):
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    return f'{timestamp}_i{ii}'


def infer_shot_mode(args):
    return 'zeroshot' if args.train_epochs == 0 else 'fullshot'


def compact_model_id(args):
    model_id_parts = args.model_id.split('_')
    pred_len = str(args.pred_len)

    if model_id_parts and model_id_parts[-1] == pred_len:
        model_id_parts = model_id_parts[:-1]

    if model_id_parts and model_id_parts[-1] == args.data:
        model_id_parts = model_id_parts[:-1]

    compact_id = '_'.join(model_id_parts).strip('_')
    return compact_id or args.model_id


def build_setting(args, ii, extra_tag=None):
    run_tag = build_run_tag(ii)
    extra_suffix = f'_{extra_tag}' if extra_tag else ''
    if args.save_dir != '.':
        shot_mode = infer_shot_mode(args)
        compact_id = compact_model_id(args)
        return f'{compact_id}{extra_suffix}_{shot_mode}_{run_tag}'
    des = f'{args.des}{extra_suffix}'
    return '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        des,
        run_tag
    )


def load_best_validation_record(args, setting):
    valid_loss_path = os.path.join(args.save_dir, args.checkpoints, setting, 'valid_loss.json')
    if not os.path.isfile(valid_loss_path):
        raise FileNotFoundError(f'Validation summary not found: {valid_loss_path}')

    with open(valid_loss_path) as f:
        record = json.load(f)

    return record['best_valid_loss'], record['best_valid_epoch']


def batch_size_search_summary_path(args):
    os.makedirs(args.save_dir, exist_ok=True)
    return os.path.join(args.save_dir, 'batch_size_search_summary.json')


def save_batch_size_search_summary(args, search_records, best_record):
    summary = {
        'data': args.data,
        'pred_len': args.pred_len,
        'model': args.model,
        'search_epochs': args.batch_size_search_epochs,
        'candidates': search_records,
        'best': best_record,
    }
    with open(batch_size_search_summary_path(args), 'w') as f:
        json.dump(summary, f, indent=2)


def save_global_best_batch_size_checkpoint(base_args, best_record, final_setting):
    source_dir = os.path.join(base_args.save_dir, base_args.checkpoints, best_record['setting'])
    target_dir = os.path.join(base_args.save_dir, base_args.checkpoints, final_setting)
    os.makedirs(target_dir, exist_ok=True)

    for filename in ('checkpoint.pth', 'valid_loss.json'):
        source_path = os.path.join(source_dir, filename)
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f'Missing search artifact: {source_path}')
        shutil.copy2(source_path, os.path.join(target_dir, filename))

    metadata = {
        'selected_batch_size': best_record['batch_size'],
        'source_setting': best_record['setting'],
        'best_valid_loss': best_record['best_valid_loss'],
        'best_valid_epoch': best_record['best_valid_epoch'],
    }
    with open(os.path.join(target_dir, 'batch_size_search_selection.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    return target_dir


def run_batch_size_search(base_args, Exp, ii):
    if base_args.task_name != 'long_term_forecast':
        raise ValueError('Batch size search is only supported for long_term_forecast.')
    if not base_args.batch_size_candidates:
        raise ValueError('Batch size search requires at least one candidate batch size.')
    if base_args.batch_size_search_epochs <= 0:
        raise ValueError('batch_size_search_epochs must be positive.')

    best_record = None
    search_records = []

    for batch_size in base_args.batch_size_candidates:
        candidate_args = copy.deepcopy(base_args)
        candidate_args.batch_size = batch_size
        candidate_args.train_epochs = candidate_args.batch_size_search_epochs

        exp = Exp(candidate_args)
        setting = build_setting(candidate_args, ii, extra_tag=f'bs{batch_size}')

        print('>>>>>>>start training with batch_size={} : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(batch_size, setting))
        exp.train(setting)

        best_valid_loss, best_valid_epoch = load_best_validation_record(candidate_args, setting)
        record = {
            'batch_size': batch_size,
            'setting': setting,
            'best_valid_loss': best_valid_loss,
            'best_valid_epoch': best_valid_epoch,
        }
        search_records.append(record)
        print('>>>>>>>validation result: batch_size={}, best_valid_loss={}, best_valid_epoch={}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(
            batch_size,
            best_valid_loss,
            best_valid_epoch,
        ))

        if best_record is None or best_valid_loss < best_record['best_valid_loss']:
            best_record = {
                **record,
                'args': candidate_args,
            }

        del exp
        torch.cuda.empty_cache()

    best_test_args = copy.deepcopy(base_args)
    best_test_args.batch_size = best_record['batch_size']
    best_test_args.train_epochs = best_test_args.batch_size_search_epochs
    final_setting = build_setting(best_test_args, ii, extra_tag=f'bestbs{best_record["batch_size"]}')
    final_checkpoint_dir = save_global_best_batch_size_checkpoint(base_args, best_record, final_setting)

    summary_best_record = {k: v for k, v in best_record.items() if k != 'args'}
    summary_best_record['final_setting'] = final_setting
    summary_best_record['final_checkpoint_dir'] = final_checkpoint_dir
    save_batch_size_search_summary(base_args, search_records, summary_best_record)

    print('>>>>>>>selected best batch_size={} with valid_loss={} : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(
        best_record['batch_size'],
        best_record['best_valid_loss'],
        final_setting,
    ))

    best_exp = Exp(best_test_args)
    print('>>>>>>>testing best setting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(final_setting))
    best_exp.test(final_setting, test=1)
    del best_exp
    torch.cuda.empty_cache()

if __name__ == '__main__':
    # fix_seed = 2021
    # random.seed(fix_seed)
    # torch.manual_seed(fix_seed)
    # np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--save_dir', type=str, default='.', help='save dir')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--channel_independence', type=int, default=0,
                        help='1: channel dependence 0: channel independence for FreTS model')
    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--batch_size_candidates', type=int, nargs='+', default=None,
                        help='candidate batch sizes for validation-based search')
    parser.add_argument('--batch_size_search_epochs', type=int, default=10,
                        help='maximum training epochs for each candidate batch size during search')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # VisionTS
    parser.add_argument('--vm_pretrained', type=int, default=1)
    parser.add_argument('--vm_ckpt', type=str, default="./ckpt/")
    parser.add_argument('--vm_arch', type=str, default='mae_base')
    parser.add_argument('--ft_type', type=str, default='ln')
    parser.add_argument('--periodicity', type=int, default=0)
    parser.add_argument('--interpolation', type=str, default='bilinear')
    parser.add_argument('--norm_const', type=float, default=0.4)
    parser.add_argument('--align_const', type=float, default=0.4)
    parser.add_argument('--rgb_mode', type=str, default='duplicate',
                        help="image rendering mode for VisionTS, options:['duplicate', 'decomposition']")
    parser.add_argument('--rgb_ma_kernel', type=int, default=5,
                        help='moving-average kernel used by decomposition rgb mode')
    parser.add_argument('--rgb_channel_scales', type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help='channel scaling factors for R/G/B when rgb_mode=decomposition')
    parser.add_argument('--rgb_dynamic_scale_mode', type=str, default='none',
                        help="dynamic rgb channel scaling mode, options:['none', 'batch', 'sample']")
    parser.add_argument('--rgb_scale_eps', type=float, default=1e-5,
                        help='epsilon used by dynamic rgb channel scaling')
    parser.add_argument('--export_rgb_vis', action='store_true', default=False,
                        help='export VisionTS decomposition RGB visualizations during test')
    parser.add_argument('--rgb_vis_max_samples', type=int, default=1,
                        help='maximum number of batch samples to export for VisionTS RGB visualization')
    parser.add_argument('--rgb_vis_max_vars', type=int, default=3,
                        help='maximum number of variables per sample to export for VisionTS RGB visualization')


    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.model == 'VisionTS' and args.periodicity <= 0:
        raise ValueError("VisionTS requires --periodicity > 0 for time-series-to-image segmentation")

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            if args.batch_size_candidates:
                run_batch_size_search(args, Exp, ii)
            else:
                exp = Exp(args)  # set experiments
                setting = build_setting(args, ii)

                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)

                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)
                torch.cuda.empty_cache()
    else:
        ii = 0
        setting = build_setting(args, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()

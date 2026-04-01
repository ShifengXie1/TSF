from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _unwrap_model(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _channel_stats(self, channel_2d):
        abs_channel = np.abs(channel_2d)
        return {
            'min': float(np.min(channel_2d)),
            'max': float(np.max(channel_2d)),
            'mean': float(np.mean(channel_2d)),
            'std': float(np.std(channel_2d)),
            'mean_abs': float(np.mean(abs_channel)),
            'max_abs': float(np.max(abs_channel)),
            'p95_abs': float(np.percentile(abs_channel, 95)),
        }

    def _save_rgb_visualization_figure(self, rendered_rgb, component_names, channel_scales, save_path):
        max_abs = float(np.max(np.abs(rendered_rgb)))
        max_abs = max(max_abs, 1e-6)
        rgb_plot = np.clip(0.5 + rendered_rgb.transpose(1, 2, 0) / (2 * max_abs), 0.0, 1.0)

        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        axes[0].imshow(rgb_plot, aspect='auto')
        axes[0].set_title('Rendered RGB')
        axes[0].axis('off')

        for idx, name in enumerate(component_names):
            im = axes[idx + 1].imshow(
                rendered_rgb[idx],
                cmap='coolwarm',
                vmin=-max_abs,
                vmax=max_abs,
                aspect='auto',
            )
            axes[idx + 1].set_title(f'{name}\nscale={channel_scales[idx]:.4g}')
            axes[idx + 1].axis('off')
            fig.colorbar(im, ax=axes[idx + 1], fraction=0.046, pad=0.04)

        fig.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

    def _export_visionts_rgb_visualization(self, batch_x, folder_path):
        if self.args.model != 'VisionTS' or not self.args.export_rgb_vis:
            return

        core_model = self._unwrap_model()
        if not hasattr(core_model, 'export_rgb_visualization'):
            return

        vis_payload = core_model.export_rgb_visualization(batch_x)
        vis_dir = os.path.join(folder_path, 'rgb_vis')
        os.makedirs(vis_dir, exist_ok=True)

        component_names = list(vis_payload['component_names'])
        resized_components = vis_payload['resized_components'].detach().cpu().numpy()
        rendered_components = vis_payload['rendered_components'].detach().cpu().numpy()
        channel_scales = vis_payload['channel_scales'].detach().cpu().numpy()

        max_samples = min(self.args.rgb_vis_max_samples, rendered_components.shape[0])
        max_vars = min(self.args.rgb_vis_max_vars, rendered_components.shape[1])

        summary = {
            'rgb_mode': vis_payload['rgb_mode'],
            'component_names': component_names,
            'rgb_channel_scales_arg': list(self.args.rgb_channel_scales),
            'rgb_dynamic_scale_mode': self.args.rgb_dynamic_scale_mode,
            'items': [],
        }

        for sample_idx in range(max_samples):
            for var_idx in range(max_vars):
                rendered_rgb = rendered_components[sample_idx, var_idx]
                resized_rgb = resized_components[sample_idx, var_idx]
                applied_scales = channel_scales[sample_idx, var_idx, :, 0, 0]

                self._save_rgb_visualization_figure(
                    rendered_rgb,
                    component_names,
                    applied_scales,
                    os.path.join(vis_dir, f'sample_{sample_idx:02d}_var_{var_idx:02d}.png'),
                )

                item = {
                    'sample_index': sample_idx,
                    'variable_index': var_idx,
                    'channel_scales': [float(scale) for scale in applied_scales],
                    'raw_stats': {},
                    'rendered_stats': {},
                }
                for comp_idx, name in enumerate(component_names):
                    item['raw_stats'][name] = self._channel_stats(resized_rgb[comp_idx])
                    item['rendered_stats'][name] = self._channel_stats(rendered_rgb[comp_idx])
                summary['items'].append(item)

        with open(os.path.join(vis_dir, 'channel_stats.json'), 'w') as f:
            json.dump(summary, f, indent=2)

    def _shot_mode(self):
        return 'zeroshot' if self.args.train_epochs == 0 else 'fullshot'

    def _build_summary_lines(self, setting, metrics, best_valid_loss, best_valid_epoch):
        mae, mse, rmse, mape, mspe = metrics
        lines = [
            f'setting: {setting}',
            f'shot_mode: {self._shot_mode()}',
            f'model_id: {self.args.model_id}',
            f'data: {self.args.data}',
            f'seq_len: {self.args.seq_len}',
            f'pred_len: {self.args.pred_len}',
            f'batch_size: {self.args.batch_size}',
            f'train_epochs: {self.args.train_epochs}',
            f'learning_rate: {self.args.learning_rate}',
            f'mse: {mse}',
            f'mae: {mae}',
            f'best_valid_loss: {best_valid_loss}',
            f'best_valid_epoch: {best_valid_epoch}',
        ]

        if self.args.model == 'VisionTS':
            lines.extend([
                f'vm_arch: {self.args.vm_arch}',
                f'periodicity: {self.args.periodicity}',
                f'rgb_mode: {self.args.rgb_mode}',
            ])
            if self.args.rgb_mode == 'decomposition':
                lines.extend([
                    f'rgb_ma_kernel: {self.args.rgb_ma_kernel}',
                    f'rgb_channel_scales: {list(self.args.rgb_channel_scales)}',
                    f'rgb_dynamic_scale_mode: {self.args.rgb_dynamic_scale_mode}',
                    f'rgb_scale_eps: {self.args.rgb_scale_eps}',
                ])

        return lines

    def _checkpoint_dir(self, setting):
        return os.path.join(self.args.save_dir, self.args.checkpoints, setting)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(vali_loader, desc='vali')):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = self._checkpoint_dir(setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            pbar = tqdm(train_loader)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pbar):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                pbar.set_description("epoch: {0} | loss: {1:.7f}".format(epoch + 1, loss.item()))
                if (i + 1) % 100 == 0:
                    tqdm.write("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    tqdm.write('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        if os.path.isfile(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
        else:
            print("Test without train!",best_model_path)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        checkpoint_dir = self._checkpoint_dir(setting)
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'checkpoint.pth')))

        valid_loss_path = os.path.join(checkpoint_dir, 'valid_loss.json')
        if os.path.isfile(valid_loss_path):
            with open(valid_loss_path) as f:
                valid_loss = json.load(f)
                best_valid_loss = valid_loss['best_valid_loss']
                best_valid_epoch = valid_loss['best_valid_epoch']
        else:
            best_valid_loss = -1
            best_valid_epoch = -1
        

        preds = []
        trues = []

        folder_path = f'{self.args.save_dir}/results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        rgb_vis_exported = False
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader, desc='test')):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if not rgb_vis_exported:
                    self._export_visionts_rgb_visualization(batch_x, folder_path)
                    rgb_vis_exported = self.args.model == 'VisionTS' and self.args.export_rgb_vis

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        summary_path = os.path.join(folder_path, 'summary.txt')
        summary_lines = self._build_summary_lines(
            setting,
            (mae, mse, rmse, mape, mspe),
            best_valid_loss,
            best_valid_epoch,
        )
        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary_lines) + '\n')

        result_log_path = os.path.join(self.args.save_dir, 'result_long_term_forecast.txt')
        with open(result_log_path, 'a') as f:
            f.write(f'{setting}\n')
            f.write(f'shot_mode:{self._shot_mode()}, pred_len:{self.args.pred_len}, mse:{mse}, mae:{mae}\n\n')

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, best_valid_loss, best_valid_epoch]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

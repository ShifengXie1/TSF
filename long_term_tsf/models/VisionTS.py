import importlib.util
import inspect
import sys
from pathlib import Path

from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_local_visionts():
    module_name = 'visionts_local'
    if module_name in sys.modules:
        return sys.modules[module_name]

    package_dir = REPO_ROOT / 'visionts'
    init_path = package_dir / '__init__.py'
    spec = importlib.util.spec_from_file_location(
        module_name,
        init_path,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load local visionts package from {init_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


VisionTS = _load_local_visionts().VisionTS

class Model(nn.Module):

    def __init__(self, config):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = config.task_name
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len

        visionts_signature = inspect.signature(VisionTS.__init__).parameters
        supports_rgb_rendering = 'rgb_mode' in visionts_signature
        supports_dynamic_rgb_scaling = 'rgb_dynamic_scale_mode' in visionts_signature and 'rgb_scale_eps' in visionts_signature
        if config.rgb_mode != 'duplicate' and not supports_rgb_rendering:
            raise RuntimeError(
                "The local VisionTS source does not support rgb_mode. "
                "Please sync the updated visionts/model.py to the server."
            )
        if config.rgb_dynamic_scale_mode != 'none' and not supports_dynamic_rgb_scaling:
            raise RuntimeError(
                "The local VisionTS source does not support rgb_dynamic_scale_mode. "
                "Please sync the updated visionts/model.py to the server."
            )

        visionts_kwargs = {}
        if supports_rgb_rendering:
            visionts_kwargs.update(
                {
                    'rgb_mode': config.rgb_mode,
                    'rgb_ma_kernel': config.rgb_ma_kernel,
                    'rgb_channel_scales': tuple(config.rgb_channel_scales),
                }
            )
        if supports_dynamic_rgb_scaling:
            visionts_kwargs.update(
                {
                    'rgb_dynamic_scale_mode': config.rgb_dynamic_scale_mode,
                    'rgb_scale_eps': config.rgb_scale_eps,
                }
            )

        self.vm = VisionTS(
            arch=config.vm_arch,
            finetune_type=config.ft_type,
            load_ckpt=config.vm_pretrained == 1,
            ckpt_dir=config.vm_ckpt,
            **visionts_kwargs,
        )

        self.vm.update_config(context_len=config.seq_len, pred_len=config.pred_len, periodicity=config.periodicity, interpolation=config.interpolation, norm_const=config.norm_const, align_const=config.align_const)


    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):

        return self.vm.forward(x_enc)


    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        raise NotImplementedError()


    def anomaly_detection(self, x_enc):
        raise NotImplementedError()

    def classification(self, x_enc, x_mark_enc):
        raise NotImplementedError()


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None

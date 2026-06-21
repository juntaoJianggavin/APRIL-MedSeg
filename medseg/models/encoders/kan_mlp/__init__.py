"""KAN/MLP编码器 / KAN / MLP encoders."""

import sys as _sys

for _stem in ('ukan_encoder', 'rolling_unet_encoder', 'unext_encoder', 'wa_ukan_encoder'):
    try:
        _mod = __import__(f'medseg.models.encoders.kan_mlp.{_stem}', fromlist=[_stem])
        _sys.modules[f'medseg.models.encoders.{_stem}'] = _mod
        globals()[_stem] = _mod
    except ImportError:
        pass

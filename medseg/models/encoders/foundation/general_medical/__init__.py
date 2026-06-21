"""General medical CLIP / SigLIP foundation encoders.
    General medical CLIP / SigLIP foundation 编码器。"""
import sys as _sys
for _stem in ('biomedclip_encoder', 'medclip_encoder', 'medsiglip_encoder'):
    try:
        _mod = __import__(f'medseg.models.encoders.foundation.general_medical.{_stem}', fromlist=[_stem])
        _sys.modules[f'medseg.models.encoders.foundation.{_stem}'] = _mod
        _sys.modules[f'medseg.models.encoders.{_stem}'] = _mod
        globals()[_stem] = _mod
    except ImportError as e:
        import warnings; warnings.warn(f'Could not import {_stem}: {e}')

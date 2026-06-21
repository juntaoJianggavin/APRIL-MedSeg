"""放射学 foundation encoders。
    Radiology foundation encoders."""
import sys as _sys
for _stem in ('raddino_encoder', 'omnirad_encoder',
              'chexzero_encoder', 'biovil_encoder'):
    try:
        _mod = __import__(f'medseg.models.encoders.foundation.radiology.{_stem}', fromlist=[_stem])
        _sys.modules[f'medseg.models.encoders.foundation.{_stem}'] = _mod
        _sys.modules[f'medseg.models.encoders.{_stem}'] = _mod
        globals()[_stem] = _mod
    except ImportError as e:
        import warnings; warnings.warn(f'Could not import {_stem}: {e}')

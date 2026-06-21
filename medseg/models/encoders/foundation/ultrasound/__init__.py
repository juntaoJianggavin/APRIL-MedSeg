"""超声基础编码器 / Ultrasound foundation encoders."""
for _stem in ('usfmae_encoder', 'ultrafedfm_encoder'):
    try:
        __import__(f"{__name__}.{_stem}")
    except ImportError:
        pass

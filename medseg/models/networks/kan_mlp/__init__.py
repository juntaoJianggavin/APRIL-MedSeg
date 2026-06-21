"""KAN / MLP / LSTM-based complete 分割 architectures。
    KAN / MLP / LSTM-based complete segmentation architectures."""

from .ukan import UKAN
from .wa_ukan import WAUKAN

__all__ = ["UKAN", "WAUKAN"]

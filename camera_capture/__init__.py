"""Independent local RGB/depth acquisition. Importing never opens a device."""
from .manager import AstraCapture, CaptureConfig, CaptureFrame, DeviceSelection

__all__ = ['AstraCapture', 'CaptureConfig', 'CaptureFrame', 'DeviceSelection']

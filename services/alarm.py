from __future__ import annotations

import math
import struct
import wave
from pathlib import Path


def ensure_alarm_wav(path: str | Path) -> Path:
    output = Path(path)
    if output.exists() and output.stat().st_size > 100:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 22050
    duration_s = 1.0
    samples: list[bytes] = []
    for index in range(int(sample_rate * duration_s)):
        second = index / sample_rate
        frequency = 880.0 if int(second * 6) % 2 == 0 else 660.0
        envelope = min(1.0, second * 20.0, (duration_s - second) * 12.0)
        value = int(14000 * envelope * math.sin(2.0 * math.pi * frequency * second))
        samples.append(struct.pack("<h", value))
    with wave.open(str(output), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"".join(samples))
    return output

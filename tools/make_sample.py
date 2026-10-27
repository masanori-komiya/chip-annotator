"""Synthetic test fixture, not the user's unavailable photograph."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from chipannotator.core import png


def sample():
    image = np.full((480, 800, 3), (42, 74, 48), np.uint8)
    cv2.rectangle(image, (140, 150), (660, 330), (65, 62, 55), -1)
    cv2.rectangle(image, (140, 150), (245, 330), (228, 230, 234), -1)
    cv2.rectangle(image, (555, 150), (660, 330), (232, 231, 225), -1)
    cv2.putText(image, '102', (320, 265), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (180, 175, 165), 3)
    return image


if __name__ == '__main__':
    target = Path(__file__).resolve().parents[1]/'samples'/'synthetic_chip.png'
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(png(cv2.cvtColor(sample(), cv2.COLOR_RGB2BGR)))
    print(target)

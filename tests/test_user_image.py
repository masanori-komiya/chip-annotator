from pathlib import Path
import pytest
import cv2
import numpy as np
from chipannotator.core import read_image, terminal_candidates, Document, export_yolo, NAMES


def test_provided_photo_candidates_and_export(tmp_path):
    source = Path(__file__).resolve().parents[1]/'samples/user_chip.png'
    if not source.exists():
        pytest.skip('実写真はローカル専用の任意テスト素材です')
    image = read_image(source)
    assert image.shape == (315,429,3)
    doc = Document(image, str(source))
    points = np.array([[54,112],[143,23],[177,33],[337,124],[338,216],[255,286],[55,181]], np.int32)
    cv2.fillPoly(doc.masks['chip'], [points], 1)
    candidates = terminal_candidates(image, doc.masks['chip'])
    for name in ('left','right'):
        assert candidates[name].sum() > 1000
        doc.masks[name] = candidates[name]
    ly, lx = np.where(candidates['left'])
    ry, rx = np.where(candidates['right'])
    assert lx.mean() < rx.mean()
    doc.confirmed = dict.fromkeys(NAMES,True)
    key, diagnostics = export_yolo(doc,tmp_path)
    assert len((tmp_path/'labels'/f'{key}.txt').read_text().splitlines()) == 3

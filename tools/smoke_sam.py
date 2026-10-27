"""Opt-in real checkpoint integration test: no mocked inference."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from chipannotator.core import Document, terminal_candidates, bbox, export_yolo, read_image
from chipannotator.sam_backend import MobileSAM
from tools.make_sample import sample


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--device', default='cuda', choices=['cuda','cpu','auto'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--image', type=Path)
    p.add_argument('--box', nargs=4, type=int)
    args = p.parse_args()
    start = time.perf_counter()
    model = MobileSAM(args.weights, args.device)
    image = read_image(args.image) if args.image else sample()
    doc = Document(image, str(args.image.resolve()) if args.image else 'synthetic_chip.png')
    doc.boxes['chip'] = args.box or [135, 145, 665, 335]
    doc.masks['chip'] = model.predict(image, doc.boxes['chip'])
    candidates = terminal_candidates(image, doc.masks['chip'])
    for name, candidate in candidates.items():
        b = bbox(candidate)
        yy, xx = np.where(candidate)
        point = [[int(xx[len(xx)//2]), int(yy[len(yy)//2]), 1]]
        doc.masks[name] = model.predict(image, b, point, doc.masks['chip'])
        doc.boxes[name], doc.points[name] = b, point
    # Exercise positive/negative points on chip too, without losing the test's terminals.
    import cv2
    distance = cv2.distanceTransform(doc.masks['chip'], cv2.DIST_L2, 3)
    y, x = np.unravel_index(distance.argmax(), distance.shape)
    refined = model.predict(image, doc.boxes['chip'], [[int(x),int(y),1],[0,0,0]])
    assert refined.any() and not refined[0,0]
    doc.confirmed = dict.fromkeys(doc.masks, True)
    key, diagnostics = export_yolo(doc, args.output)
    report = dict(device=model.device, elapsed_seconds=round(time.perf_counter()-start, 3),
                  pixels={k:int(m.sum()) for k,m in doc.masks.items()},
                  diagnostics=diagnostics, key=key)
    (args.output/'sam_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

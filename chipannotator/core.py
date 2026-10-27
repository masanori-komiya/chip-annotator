from __future__ import annotations

import hashlib
import json
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

NAMES = ('chip', 'left', 'right')
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'画像を読み込めません: {path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def png(array):
    ok, data = cv2.imencode('.png', array)
    if not ok:
        raise ValueError('PNGの作成に失敗しました')
    return data.tobytes()


def bbox(mask):
    y, x = np.where(mask)
    if not len(x):
        raise ValueError('マスクが空です。矩形またはブラシで対象を指定してください。')
    return [int(x.min()), int(y.min()), int(x.max()), int(y.max())]


@dataclass
class Document:
    image: np.ndarray
    source: str
    masks: dict = field(default_factory=dict)
    boxes: dict = field(default_factory=dict)
    points: dict = field(default_factory=lambda: {k: [] for k in NAMES})
    confirmed: dict = field(default_factory=lambda: {k: False for k in NAMES})

    terminals_skipped: bool = False

    def __post_init__(self):
        for name in NAMES:
            self.masks.setdefault(name, np.zeros(self.image.shape[:2], np.uint8))

    def changed(self, name):
        self.confirmed[name] = False
        if name == 'chip':
            self.confirmed['left'] = self.confirmed['right'] = False

    @property
    def annotation_names(self):
        return ('chip',) if self.terminals_skipped else NAMES

    @property
    def ready_for_export(self):
        return all(self.confirmed.get(k) for k in self.annotation_names)

    def metadata(self):
        return dict(version=1, source=self.source, shape=list(self.image.shape),
                    boxes=self.boxes, points=self.points, confirmed=self.confirmed,
                    terminals_skipped=self.terminals_skipped)


def save_project(doc, path):
    """Self contained, atomic archive; never pickle or depend on source paths."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    try:
        with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('project.json', json.dumps(doc.metadata(), ensure_ascii=False))
            z.writestr('image.png', png(cv2.cvtColor(doc.image, cv2.COLOR_RGB2BGR)))
            for name in NAMES:
                z.writestr(f'{name}.png', png((doc.masks[name] > 0).astype(np.uint8) * 255))
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def load_project(path):
    with zipfile.ZipFile(path) as z:
        # Read named entries only, never extract paths from an archive.
        if sum(i.file_size for i in z.infolist()) > 1024 ** 3:
            raise ValueError('プロジェクトが大きすぎます (展開上限1GB)')
        meta = json.loads(z.read('project.json'))
        if meta.get('version') != 1:
            raise ValueError('未対応のプロジェクト形式です')
        image = cv2.imdecode(np.frombuffer(z.read('image.png'), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('プロジェクトの画像が破損しています')
        doc = Document(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), str(meta['source']))
        skipped = meta.get('terminals_skipped', False)
        if not isinstance(skipped, bool):
            raise ValueError('端子スキップ状態が不正です')
        doc.terminals_skipped = skipped
        for name in NAMES:
            mask = cv2.imdecode(np.frombuffer(z.read(f'{name}.png'), np.uint8), 0)
            if mask is None or mask.shape != image.shape[:2]:
                raise ValueError('マスクのサイズが不正です')
            doc.masks[name] = (mask > 0).astype(np.uint8)
            points = meta.get('points', {}).get(name, [])
            if any(len(p) != 3 or p[2] not in (0, 1) or
                   not (0 <= p[0] < image.shape[1] and 0 <= p[1] < image.shape[0]) for p in points):
                raise ValueError('点プロンプトが不正です')
            doc.points[name] = points
            doc.confirmed[name] = bool(meta.get('confirmed', {}).get(name, False))
            box = meta.get('boxes', {}).get(name)
            if box is not None:
                if len(box) != 4 or not (0 <= box[0] < box[2] < image.shape[1] and
                                                       0 <= box[1] < box[3] < image.shape[0]):
                    raise ValueError('矩形プロンプトが不正です')
                doc.boxes[name] = box
        return doc


def terminal_candidates(image, chip):
    """Brightness + low saturation + end position + component compactness.

    Ends are defined by the chip's principal axis, ordered by screen x (then y).
    Return masks, never fabricate a terminal when the side contains no white area.
    """
    chip = (chip > 0).astype(np.uint8)
    yy, xx = np.where(chip)
    if len(xx) < 30:
        raise ValueError('チップマスクが小さすぎます')
    coords = np.column_stack((xx, yy)).astype(np.float64)
    center = coords.mean(0)
    _, vectors = np.linalg.eigh(np.cov(coords.T))
    axis = vectors[:, -1]
    if axis[0] < -1e-8 or (abs(axis[0]) < 1e-8 and axis[1] < 0):
        axis = -axis
    projected = (coords - center) @ axis
    lo, hi = np.quantile(projected, [0.01, 0.99])
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    values = hsv[:, :, 2][chip > 0]
    threshold = max(100, float(np.percentile(values, 65)))
    white = ((hsv[:, :, 2] >= threshold) & (hsv[:, :, 1] < 110) & (chip > 0)).astype(np.uint8)
    projection = np.zeros(chip.shape, np.float32)
    projection[yy, xx] = projected
    result = {}
    for name, region in [('left', projection <= lo + (hi-lo)*0.38),
                         ('right', projection >= hi - (hi-lo)*0.38)]:
        binary = white * region.astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) * chip
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        best, best_score = None, -1
        for i in range(1, count):
            x, y, w, h, area = stats[i]
            if area < max(5, len(xx)*0.0005):
                continue
            component = labels == i
            brightness = float(hsv[:, :, 2][component].mean()) / 255
            compactness = area / (w*h)
            score = area * brightness * (0.5 + compactness)
            if score > best_score:
                best, best_score = component.astype(np.uint8), score
        if best is None:
            raise ValueError(f'{name}端子の白い候補がありません。端子を選択し矩形・ブラシで指定してください。')
        result[name] = best
    return result


def polygon(mask):
    """One YOLO polygon per instance. Bridge disconnected external contours.

    Holes cannot be represented by the YOLO simple polygon format. Exact masks
    remain separate PNGs; callers receive a rasterized IoU and topology warning.
    """
    binary = (mask > 0).astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        raise ValueError('空マスクは書き出せません')
    external = [c[:, 0, :] for i, c in enumerate(contours)
                if hierarchy[0, i, 3] == -1 and len(c) >= 3 and cv2.contourArea(c) > 0]
    if not external:
        raise ValueError('面積のある輪郭がありません。1ピクセルの線・点を修正してください。')
    external.sort(key=lambda c: abs(cv2.contourArea(c)), reverse=True)
    merged = external.pop(0)
    for other in external:
        # Chunked nearest-point search bounds temporary memory on detailed masks.
        best = (float('inf'), 0, 0)
        for start in range(0, len(merged), 256):
            d = ((merged[start:start+256, None].astype(np.float64) - other[None]) ** 2).sum(2)
            a, b = np.unravel_index(d.argmin(), d.shape)
            if d[a, b] < best[0]:
                best = (d[a, b], start + a, b)
        _, a, b = best
        loop = np.concatenate((other[b:], other[:b+1]))
        merged = np.concatenate((merged[:a+1], loop, merged[a:]))
    raster = np.zeros_like(binary)
    cv2.fillPoly(raster, [merged.astype(np.int32)], 1)
    union = np.count_nonzero(raster | binary)
    iou = float(np.count_nonzero(raster & binary) / union) if union else 1.0
    holes = int(sum(h[3] >= 0 for h in hierarchy[0]))
    return merged, dict(iou=iou, holes=holes, components=int(sum(h[3] == -1 for h in hierarchy[0])))


def document_key(source):
    p = Path(source)
    digest = hashlib.sha256(str(p.resolve()).encode('utf-8')).hexdigest()[:10]
    return f'{p.stem}_{digest}'


def export_yolo(doc, root):
    if not doc.ready_for_export:
        raise ValueError('対象を確定してください。端子なしの場合は端子工程をスキップできます。')
    h, w = doc.image.shape[:2]
    rows, polygons, diagnostics = [], {}, {}
    for name, cls in [('chip', 0), ('left', 1), ('right', 1)]:
        if name not in doc.annotation_names:
            continue
        points, info = polygon(doc.masks[name])
        normalized = points.astype(np.float64) / [w, h]
        rows.append(str(cls) + ' ' + ' '.join(f'{v:.8f}' for v in normalized.ravel()))
        polygons[name] = dict(class_id=cls, pixel=points.tolist(), normalized=normalized.tolist())
        diagnostics[name] = info
    root = Path(root)
    key = document_key(doc.source)
    # Individual files are replaced atomically; publish the label last.
    for folder in ('images', 'labels', 'masks', 'polygons', 'projects'):
        (root/folder).mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        mask = doc.masks[name] if name in doc.annotation_names else np.zeros((h, w), np.uint8)
        atomic_bytes(root/'masks'/f'{key}_{name}.png', png(mask.astype(np.uint8)*255))
    atomic_bytes(root/'images'/f'{key}.png', png(cv2.cvtColor(doc.image, cv2.COLOR_RGB2BGR)))
    atomic_bytes(root/'polygons'/f'{key}.json', json.dumps(dict(polygons=polygons, diagnostics=diagnostics, terminals_skipped=doc.terminals_skipped), indent=2).encode())
    save_project(doc, root/'projects'/f'{key}.chipproj')
    if not (root/'data.yaml').exists():
        config = ('path: ' + json.dumps(root.resolve().as_posix(), ensure_ascii=False) +
                  '\ntrain: images\nval: images\nnames:\n  0: chip\n  1: terminal\n')
        atomic_bytes(root/'data.yaml', config.encode('utf-8'))
    atomic_bytes(root/'labels'/f'{key}.txt', ('\n'.join(rows)+'\n').encode())
    return key, diagnostics


def atomic_bytes(path, data):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    try:
        temp.write_bytes(data)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)

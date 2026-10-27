import json
import zipfile
import cv2
import numpy as np
import pytest

from chipannotator.core import (Document, NAMES, bbox, terminal_candidates,
    save_project, load_project, export_yolo, polygon, document_key, read_image)
from chipannotator.sam_backend import MobileSAM, ModelError
from tools.make_sample import sample


def document():
    doc = Document(sample(), '合成チップ.png')
    doc.masks['chip'][150:331, 140:661] = 1
    candidates = terminal_candidates(doc.image, doc.masks['chip'])
    doc.masks.update(candidates)
    doc.boxes['chip'] = [140, 150, 660, 330]
    doc.points['chip'] = [[400, 240, 1], [30, 30, 0]]
    return doc


def test_white_terminals_and_overlap():
    doc = document()
    for name, expected in [('left', [140, 150, 245, 330]), ('right', [555, 150, 660, 330])]:
        assert bbox(doc.masks[name]) == expected
        assert np.all(doc.masks[name] <= doc.masks['chip'])


@pytest.mark.parametrize('angle', [30, 80, 90, 135])
def test_rotated_candidates(angle):
    doc = document()
    transform = cv2.getRotationMatrix2D((400, 240), angle, .6)
    image = cv2.warpAffine(doc.image, transform, (800, 480))
    chip = cv2.warpAffine(doc.masks['chip'], transform, (800, 480), flags=cv2.INTER_NEAREST)
    candidates = terminal_candidates(image, chip)
    assert candidates['left'].sum() > 3000
    assert candidates['right'].sum() > 3000
    assert not np.any(candidates['left'] & candidates['right'])


def test_dark_chip_no_fabricated_candidate():
    doc = document()
    doc.image[:] = 20
    with pytest.raises(ValueError, match='白い候補'):
        terminal_candidates(doc.image, doc.masks['chip'])


def test_roundtrip_portable_unicode(tmp_path):
    doc = document()
    doc.confirmed['chip'] = True
    path = tmp_path/'日本語'/'作業.chipproj'
    save_project(doc, path)
    loaded = load_project(path)
    assert np.array_equal(loaded.image, doc.image)
    assert loaded.boxes == doc.boxes
    assert loaded.points == doc.points
    assert loaded.confirmed == doc.confirmed
    for name in NAMES:
        assert np.array_equal(loaded.masks[name], doc.masks[name])
    # Repeated saves replace existing state, and don't leave partial files.
    doc.masks['left'][:] = 0
    save_project(doc, path)
    assert load_project(path).masks['left'].sum() == 0
    assert not path.with_name(path.name+'.tmp').exists()


def test_yolo_three_rows_and_exact_masks(tmp_path):
    doc = document()
    doc.confirmed = dict.fromkeys(NAMES, True)
    key, report = export_yolo(doc, tmp_path)
    rows = (tmp_path/'labels'/f'{key}.txt').read_text().splitlines()
    assert len(rows) == 3
    assert [int(r.split()[0]) for r in rows] == [0, 1, 1]
    for name, row in zip(NAMES, rows):
        values = np.array([float(v) for v in row.split()[1:]])
        assert len(values) >= 6 and len(values) % 2 == 0
        assert np.all((values >= 0) & (values <= 1))
        points = np.rint(values.reshape(-1, 2)*[800, 480]).astype(np.int32)
        raster = np.zeros((480, 800), np.uint8)
        cv2.fillPoly(raster, [points], 1)
        assert np.array_equal(raster, doc.masks[name])
        saved = cv2.imdecode(np.fromfile(str(tmp_path/'masks'/f'{key}_{name}.png'), np.uint8), 0)
        assert np.array_equal(saved > 0, doc.masks[name] > 0)
        assert report[name]['iou'] == 1
    assert np.array_equal(read_image(tmp_path/'images'/f'{key}.png'), doc.image)
    assert document_key('a/foo.png') != document_key('b/foo.png')
    assert tmp_path.resolve().as_posix() in (tmp_path/'data.yaml').read_text(encoding='utf-8')
    (tmp_path/'data.yaml').write_text('custom split configuration', encoding='utf-8')
    export_yolo(doc, tmp_path)
    assert (tmp_path/'data.yaml').read_text() == 'custom split configuration'


def test_unconfirmed_and_empty_block_export(tmp_path):
    doc = document()
    with pytest.raises(ValueError, match='確定'):
        export_yolo(doc, tmp_path)
    doc.confirmed = dict.fromkeys(NAMES, True)
    doc.masks['left'][:] = 0
    with pytest.raises(ValueError, match='空マスク'):
        export_yolo(doc, tmp_path)
    assert not (tmp_path/'labels').exists()


def test_edit_chip_invalidates_terminals():
    doc = document()
    doc.confirmed = dict.fromkeys(NAMES, True)
    doc.changed('left')
    assert doc.confirmed == dict(chip=True, left=False, right=True)
    doc.changed('chip')
    assert not any(doc.confirmed.values())


def test_polygon_holes_and_disconnected():
    mask = np.zeros((100, 100), np.uint8)
    mask[5:30, 5:30] = 1
    mask[50:80, 50:80] = 1
    mask[55:60, 55:60] = 0
    points, report = polygon(mask)
    assert report['holes'] == 1
    assert report['components'] == 2
    assert 0.8 < report['iou'] < 1
    assert len(points) >= 8


def test_missing_model_actionable(tmp_path):
    with pytest.raises(ModelError, match='setup_model.bat'):
        MobileSAM(tmp_path/'missing.pt').load()


def test_bad_archive_and_mask_shape(tmp_path):
    doc = document()
    p = tmp_path/'bad.chipproj'
    save_project(doc, p)
    with zipfile.ZipFile(p, 'a') as z:
        meta = doc.metadata()
        meta['version'] = 900
        with pytest.warns(UserWarning, match='Duplicate name'):
            z.writestr('project.json', json.dumps(meta))
    with pytest.raises(ValueError, match='未対応'):
        load_project(p)


def test_chip_only_roundtrip_and_overwrite_previous_terminals(tmp_path):
    doc = document()
    doc.confirmed = dict.fromkeys(NAMES, True)
    key, _ = export_yolo(doc, tmp_path)
    doc.terminals_skipped = True
    doc.confirmed['left'] = doc.confirmed['right'] = False
    assert doc.ready_for_export
    key, diagnostics = export_yolo(doc, tmp_path)
    rows = (tmp_path/'labels'/f'{key}.txt').read_text().splitlines()
    assert len(rows) == 1 and rows[0].startswith('0 ')
    assert set(diagnostics) == {'chip'}
    polygons = json.loads((tmp_path/'polygons'/f'{key}.json').read_text())
    assert polygons['terminals_skipped'] is True
    assert set(polygons['polygons']) == {'chip'}
    for name in ('left', 'right'):
        data = np.fromfile(str(tmp_path/'masks'/f'{key}_{name}.png'), np.uint8)
        assert not cv2.imdecode(data, 0).any()
    restored = load_project(tmp_path/'projects'/f'{key}.chipproj')
    assert restored.terminals_skipped and restored.ready_for_export
    assert restored.masks['left'].any()  # draft retained only inside project
    restored.terminals_skipped = False
    assert not restored.ready_for_export


def test_chip_only_requires_chip_confirmation(tmp_path):
    doc = document()
    doc.terminals_skipped = True
    assert not doc.ready_for_export
    with pytest.raises(ValueError, match='確定'):
        export_yolo(doc, tmp_path)
    doc.confirmed['chip'] = True
    doc.changed('chip')
    assert doc.terminals_skipped and not doc.ready_for_export


def test_old_project_without_skip_flag_defaults_to_terminals_required(tmp_path):
    doc = document()
    path = tmp_path/'old.chipproj'
    save_project(doc, path)
    with zipfile.ZipFile(path) as z:
        entries = {name: z.read(name) for name in z.namelist()}
    metadata = json.loads(entries['project.json'])
    metadata.pop('terminals_skipped')
    entries['project.json'] = json.dumps(metadata).encode()
    with zipfile.ZipFile(path, 'w') as z:
        for name, data in entries.items():
            z.writestr(name, data)
    assert not load_project(path).terminals_skipped

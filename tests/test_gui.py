import time
import numpy as np
import pytest
from PySide6.QtCore import Qt, QPointF
from PySide6.QtWidgets import QApplication, QMessageBox, QFileDialog
from PySide6.QtTest import QTest
from chipannotator.gui import Window
from chipannotator.core import Document, NAMES, load_project
from tools.make_sample import sample


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app):
    w = Window()
    w.install(Document(sample(), 'synthetic.png'))
    w.show()
    app.processEvents()
    yield w
    w.dirty = False
    w.close()


def test_mouse_rectangle_brush_erase_undo(window, app):
    c = window.canvas
    start = c.mapFromScene(QPointF(140, 150))
    end = c.mapFromScene(QPointF(660, 330))
    QTest.mousePress(c.viewport(), Qt.LeftButton, pos=start)
    QTest.mouseMove(c.viewport(), end, 30)
    QTest.mouseRelease(c.viewport(), Qt.LeftButton, pos=end)
    assert len(window.doc.boxes['chip']) == 4
    window.rectangle_mask()
    assert window.doc.masks['chip'].sum() > 80000
    before = window.doc.masks['chip'].copy()
    window.tools.setCurrentIndex(4)
    mid = c.mapFromScene(QPointF(400, 240))
    QTest.mouseClick(c.viewport(), Qt.LeftButton, pos=mid)
    assert window.doc.masks['chip'].sum() < before.sum()
    window.undo()
    assert np.array_equal(before, window.doc.masks['chip'])
    window.confirm()
    assert window.doc.confirmed['chip']
    window.tools.setCurrentIndex(1)
    QTest.mouseClick(c.viewport(), Qt.LeftButton, pos=mid)
    assert window.doc.points['chip'][-1][2] == 1
    assert not window.doc.confirmed['chip']


def test_independent_manual_masks_and_project(window, tmp_path):
    for index, box in enumerate(([140,150,660,330], [140,150,245,330], [555,150,660,330])):
        window.target.setCurrentIndex(index)
        window.box_changed(list(box))
        window.rectangle_mask()
        window.confirm()
    assert all(window.doc.confirmed.values())
    window.project_path = tmp_path/'gui.chipproj'
    assert window.save()
    loaded = load_project(window.project_path)
    for k in NAMES:
        assert np.array_equal(loaded.masks[k], window.doc.masks[k])
    assert all(loaded.confirmed.values())


def test_async_result_on_gui_thread(window, app):
    result = []
    window.run_task(lambda progress: 42, result.append)
    deadline = time.monotonic()+5
    while time.monotonic() < deadline and (window.task.isRunning() or not result):
        app.processEvents()
        QTest.qWait(10)
    app.processEvents()
    assert result == [42]
    assert window.canvas.isEnabled()
    assert window.statusBar().currentMessage().startswith('処理完了')


def wait_task(window, app):
    deadline = time.monotonic()+10
    while time.monotonic() < deadline and window.task.isRunning():
        app.processEvents()
        QTest.qWait(10)
    app.processEvents()
    assert not window.task.isRunning()


class FakeModel:
    device = 'test'
    def load(self):
        pass
    def predict(self, image, box, points=None, restrict=None):
        mask = np.zeros(image.shape[:2], np.uint8)
        x1,y1,x2,y2 = box
        mask[y1:y2+1,x1:x2+1] = 1
        if restrict is not None:
            mask &= restrict
        return mask


def test_terminal_worker_preserves_chip_and_independent_confirmation(window, app, monkeypatch):
    monkeypatch.setattr(window, 'get_backend', lambda: FakeModel())
    window.doc.boxes['chip'] = [140,150,660,330]
    window.rectangle_mask()
    window.confirm()
    chip = window.doc.masks['chip'].copy()
    window.terminals()
    wait_task(window, app)
    assert np.array_equal(chip, window.doc.masks['chip'])
    assert window.doc.confirmed == dict(chip=True,left=False,right=False)
    assert window.doc.masks['left'].any() and window.doc.masks['right'].any()
    window.target.setCurrentIndex(1)
    window.confirm()
    assert window.doc.confirmed == dict(chip=True,left=True,right=False)


@pytest.mark.parametrize('skip_terminals', [False, True])
def test_batch_drafts_resume_skip_and_export(window, app, tmp_path, monkeypatch, skip_terminals):
    from chipannotator.core import png, save_project
    import cv2
    folder = tmp_path/'画像'
    folder.mkdir()
    for name in ('a.png', 'b.png'):
        (folder/name).write_bytes(png(cv2.cvtColor(sample(), cv2.COLOR_RGB2BGR)))
    window.folder = folder
    for path in sorted(folder.iterdir()):
        window.list.addItem(path.name)
        window.list.item(window.list.count()-1).setData(Qt.UserRole, str(path))
    window.doc.boxes['chip'] = [140,150,660,330]
    window.skip_terminals.setChecked(skip_terminals)
    window.dirty = False
    monkeypatch.setattr(window, 'get_backend', lambda: FakeModel())
    monkeypatch.setattr(QMessageBox, 'question', lambda *a,**k: QMessageBox.Yes)
    monkeypatch.setattr(QMessageBox, 'information', lambda *a,**k: QMessageBox.Ok)
    window.batch()
    wait_task(window, app)
    drafts = sorted((folder/'.chipannotator').glob('*.chipproj'))
    assert len(drafts) == 2
    assert not any(load_project(drafts[0]).confirmed.values())
    assert load_project(drafts[0]).terminals_skipped == skip_terminals
    if skip_terminals:
        assert not load_project(drafts[0]).masks['left'].any()
    stamps = [p.stat().st_mtime_ns for p in drafts]
    window.batch()
    wait_task(window, app)
    assert [p.stat().st_mtime_ns for p in drafts] == stamps
    doc = load_project(drafts[0])
    doc.confirmed = dict.fromkeys(NAMES, True)
    save_project(doc, drafts[0])
    output = tmp_path/'dataset'
    output.mkdir()
    choices = iter([str(folder/'.chipannotator'), str(output)])
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', lambda *a,**k: next(choices))
    window.batch_export()
    wait_task(window, app)
    labels = list((output/'labels').glob('*.txt'))
    assert len(labels) == 1
    assert len(labels[0].read_text().splitlines()) == (1 if skip_terminals else 3)
    assert 'SKIP' in (output/'export_report.txt').read_text(encoding='utf-8')


def test_skip_terminals_checkbox_undo_and_resume(window, tmp_path):
    window.doc.boxes['chip'] = [140,150,660,330]
    window.rectangle_mask()
    window.confirm()
    assert not window.doc.ready_for_export
    window.skip_terminals.setChecked(True)
    assert window.doc.terminals_skipped and window.doc.ready_for_export
    assert not window.terminals_button.isEnabled()
    assert not window.target.model().item(1).isEnabled()
    assert 'スキップ' in window.state.text()
    window.undo()
    assert not window.skip_terminals.isChecked()
    assert window.terminals_button.isEnabled()
    assert not window.doc.ready_for_export
    window.skip_terminals.setChecked(True)
    window.project_path = tmp_path/'chip_only.chipproj'
    assert window.save()
    window.install(load_project(window.project_path), window.project_path)
    assert window.skip_terminals.isChecked() and window.doc.ready_for_export
    window.skip_terminals.setChecked(False)
    assert not window.doc.ready_for_export


def test_skipping_hides_preserved_terminal_drafts(window):
    window.target.setCurrentIndex(1)
    window.doc.masks['left'][10:20,10:20] = 1
    window.skip_terminals.setChecked(True)
    assert window.active == 'chip'
    assert window.doc.masks['left'].sum() == 100
    window.skip_terminals.setChecked(False)
    assert window.doc.masks['left'].sum() == 100
    assert not window.doc.confirmed['left']

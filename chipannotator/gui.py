from __future__ import annotations

import copy
import os
import logging
import threading
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QThread, QTimer, QSettings, QPointF
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QAction, QKeySequence, QFont, QFontDatabase
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QFileDialog, QMessageBox, QComboBox, QSpinBox,
    QLabel, QListWidget, QSplitter, QGraphicsView, QGraphicsScene, QDialog,
    QFormLayout, QLineEdit, QDialogButtonBox, QInputDialog, QCheckBox)

from .core import (Document, NAMES, EXTENSIONS, read_image, bbox, terminal_candidates,
                   save_project, load_project, export_yolo, document_key)
from .sam_backend import MobileSAM

ROOT = Path(__file__).resolve().parents[1]
COLORS = {'chip': (30, 180, 255), 'left': (255, 170, 40), 'right': (200, 80, 230)}
DISPLAY = {'chip': 'チップ', 'left': '左端子', 'right': '右端子'}


class Task(QThread):
    result = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.result.emit(self.fn(self.progress.emit))
        except Exception as e:
            logging.exception('Task failed')
            self.failed.emit(str(e))


class Canvas(QGraphicsView):
    box_drawn = Signal(object)
    point_added = Signal(object)
    stroke_started = Signal()
    painted = Signal(object, object, bool)
    stroke_finished = Signal()

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.pixmap = self.scene().addPixmap(QPixmap())
        self.overlay = self.scene().addPixmap(QPixmap())
        self.overlay.setZValue(1)
        self.setBackgroundBrush(QColor('#171d28'))
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.tool = 'box'
        self.shape = (0, 0)
        self.start = self.last = None
        self.rubber = None
        self.brush_size = 16

    def set_image(self, image):
        h, w = image.shape[:2]
        self.shape = (h, w)
        self.pixmap.setPixmap(QPixmap.fromImage(QImage(image.data, w, h, image.strides[0], QImage.Format_RGB888).copy()))
        self.scene().setSceneRect(0, 0, w, h)
        self.fit()

    def fit(self):
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def show_masks(self, doc, active):
        h, w = doc.image.shape[:2]
        rgba = np.zeros((h, w, 4), np.uint8)
        for name in [k for k in NAMES if k != active] + [active]:
            if doc.terminals_skipped and name != 'chip':
                continue
            inside = doc.masks[name] > 0
            rgba[inside, :3] = COLORS[name]
            rgba[inside, 3] = 115 if name == active else 55
        q = QImage(rgba.data, w, h, rgba.strides[0], QImage.Format_RGBA8888).copy()
        painter = QPainter(q)
        painter.setPen(QPen(QColor('yellow'), 2))
        if active in doc.boxes:
            x1, y1, x2, y2 = doc.boxes[active]
            painter.drawRect(x1, y1, x2-x1, y2-y1)
        for x, y, label in doc.points[active]:
            painter.setPen(QPen(QColor('#00ff70' if label else '#ff4050'), 2))
            painter.drawEllipse(QPointF(x, y), 4, 4)
            painter.drawText(QPointF(x+5, y-5), '+' if label else '−')
        painter.end()
        self.overlay.setPixmap(QPixmap.fromImage(q))

    def position(self, event):
        p = self.mapToScene(event.position().toPoint())
        h, w = self.shape
        if not w:
            return None
        return (min(w-1, max(0, int(p.x()))), min(h-1, max(0, int(p.y()))))

    def mousePressEvent(self, event):
        if self.tool == 'pan' or event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            return super().mousePressEvent(event)
        p = self.position(event)
        if p is None or event.button() not in (Qt.LeftButton, Qt.RightButton):
            return
        self.start = self.last = p
        if self.tool == 'box':
            self.rubber = self.scene().addRect(p[0], p[1], 0, 0, QPen(QColor('yellow'), 1))
            self.rubber.setZValue(3)
        elif self.tool in ('positive', 'negative'):
            label = int(self.tool == 'positive' and event.button() != Qt.RightButton)
            self.point_added.emit([p[0], p[1], label])
            self.start = None
        elif self.tool in ('brush', 'erase'):
            self.stroke_started.emit()
            self.painted.emit(p, p, self.tool == 'erase')

    def mouseMoveEvent(self, event):
        if self.tool == 'pan':
            return super().mouseMoveEvent(event)
        if self.start is None:
            return
        p = self.position(event)
        if self.rubber:
            x, y = self.start
            self.rubber.setRect(min(x, p[0]), min(y, p[1]), abs(x-p[0]), abs(y-p[1]))
        elif self.tool in ('brush', 'erase'):
            self.painted.emit(self.last, p, self.tool == 'erase')
        self.last = p

    def mouseReleaseEvent(self, event):
        if self.tool == 'pan':
            return super().mouseReleaseEvent(event)
        if self.rubber is not None:
            p = self.position(event)
            x, y = self.start
            box = [min(x, p[0]), min(y, p[1]), max(x, p[0]), max(y, p[1])]
            self.scene().removeItem(self.rubber)
            self.rubber = None
            if box[2]-box[0] >= 2 and box[3]-box[1] >= 2:
                self.box_drawn.emit(box)
        elif self.start and self.tool in ('brush', 'erase'):
            self.stroke_finished.emit()
        self.start = self.last = None

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            factor = 1.2 if event.angleDelta().y() > 0 else 1/1.2
            scale = self.transform().m11()*factor
            if 0.01 < scale < 40:
                self.scale(factor, factor)
        else:
            super().wheelEvent(event)


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Chip Annotator | チップ・端子アノテーション')
        self.resize(1440, 920)
        self.settings = QSettings('ChipAnnotator', 'Desktop')
        self.doc = None
        self.project_path = None
        self.dirty = False
        self.undo_stack = []
        self.backend = None
        self.task = None
        self.cancel = threading.Event()
        self.folder = None
        self.output_root = None
        self.active = 'chip'
        self.canvas = Canvas()
        self.target = QComboBox()
        self.target.addItems([DISPLAY[k] for k in NAMES])
        self.target.currentIndexChanged.connect(self.select_target)
        self.skip_terminals = QCheckBox('端子なし（端子工程をスキップ）')
        self.skip_terminals.toggled.connect(self.set_terminals_skipped)
        self.tools = QComboBox()
        self.tool_ids = ['box', 'positive', 'negative', 'brush', 'erase', 'pan']
        self.tools.addItems(['矩形', '正点 ＋', '負点 −', 'ブラシ', '消しゴム', '移動'])
        self.tools.currentIndexChanged.connect(self.change_tool)
        self.size = QSpinBox()
        self.size.setRange(1, 300)
        self.size.setValue(16)
        self.size.setSuffix(' px')
        self.size.valueChanged.connect(lambda v: setattr(self.canvas, 'brush_size', v))
        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(lambda i: self.open_path(Path(i.data(Qt.UserRole))))
        self.state = QLabel('画像を開いてください')
        self.state.setWordWrap(True)
        self.device_label = QLabel('SAM: 未読込')
        self.device_label.setWordWrap(True)
        self.help = QLabel('① 矩形を描く → SAM提案\n② 点を追加 → SAM再提案\n③ ブラシで修正 → 確定\n④ 左右候補 → 各端子を修正・確定\n⑤ YOLO書き出し\n\nCtrl+ホイール: 拡大縮小\n移動ツール: ドラッグ\nCtrl+Z: 戻す / Ctrl+S: 保存\n点追加後はSAMボタンで反映\nブラシ後の再提案は手修正を置換')
        self.help.setWordWrap(True)
        left = QWidget()
        layout = QVBoxLayout(left)
        layout.addWidget(QLabel('画像一覧（ダブルクリックで開く）'))
        layout.addWidget(self.list, 1)
        layout.addWidget(self.state)
        layout.addWidget(self.device_label)
        layout.addWidget(self.help)
        left.setMinimumWidth(250)
        right = QWidget()
        rl = QVBoxLayout(right)
        bar = QHBoxLayout()
        for label, widget in [('編集対象', self.target), ('ツール', self.tools), ('ブラシ径', self.size)]:
            bar.addWidget(QLabel(label))
            bar.addWidget(widget)
        rl.addLayout(bar)
        rl.addWidget(self.skip_terminals)
        rl.addWidget(self.canvas)
        actions = QHBoxLayout()
        for text, slot in [('SAM提案 / 再提案', self.propose), ('矩形を手動マスク化', self.rectangle_mask),
                           ('この対象を確定', self.confirm), ('左右候補＋SAM', self.terminals),
                           ('戻す', self.undo), ('点を消去', self.clear_points)]:
            b = QPushButton(text)
            b.clicked.connect(slot)
            if text == '左右候補＋SAM':
                self.terminals_button = b
            actions.addWidget(b)
        rl.addLayout(actions)
        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([280, 1100])
        self.setCentralWidget(splitter)
        self.controls = [left, right]
        toolbar = self.addToolBar('ファイル')
        self.file_actions = []
        for text, slot, shortcut in [
            ('画像を開く', self.open_image, 'Ctrl+O'), ('フォルダを開く', self.open_folder, None),
            ('再開', self.resume, None), ('途中保存', self.save, 'Ctrl+S'),
            ('名前を付けて保存', lambda: self.save(True), None),
            ('YOLO書き出し', self.export, None), ('前へ', lambda: self.navigate(-1), None),
            ('次へ', lambda: self.navigate(1), None), ('一括提案', self.batch, None),
            ('一括書き出し', self.batch_export, None),
            ('モデル設定', self.configure, None), ('表示を合わせる', self.canvas.fit, 'F')]:
            action = QAction(text, self)
            action.triggered.connect(slot)
            if shortcut:
                action.setShortcut(QKeySequence(shortcut))
            toolbar.addAction(action)
            self.file_actions.append(action)
        stop = QAction('一括処理を停止', self)
        stop.triggered.connect(self.cancel.set)
        toolbar.addAction(stop)
        undo = QAction(self)
        undo.setShortcut(QKeySequence.Undo)
        undo.triggered.connect(self.undo)
        self.addAction(undo)
        self.canvas.box_drawn.connect(self.box_changed)
        self.canvas.point_added.connect(self.point_changed)
        self.canvas.stroke_started.connect(self.snapshot)
        self.canvas.painted.connect(self.paint)
        self.canvas.stroke_finished.connect(self.refresh)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.autosave)
        self.timer.start(60000)
        self.statusBar().showMessage('準備完了。モデル導入は setup_model.bat。手動編集はモデルなしで利用可能。')

    def error(self, message):
        QMessageBox.warning(self, '操作を完了できません', str(message))

    def safe(self, fn):
        try:
            return fn()
        except Exception as e:
            logging.exception('Operation failed')
            self.error(e)

    def select_target(self, index):
        self.active = NAMES[index]
        self.refresh()

    def set_terminals_skipped(self, skipped):
        if not self.doc or self.doc.terminals_skipped == skipped:
            return
        self.snapshot()
        self.doc.terminals_skipped = skipped
        self.doc.confirmed['left'] = self.doc.confirmed['right'] = False
        self.dirty = True
        self.refresh()

    def change_tool(self, index):
        self.canvas.tool = self.tool_ids[index]
        self.canvas.setDragMode(QGraphicsView.ScrollHandDrag if index == 5 else QGraphicsView.NoDrag)

    def snapshot(self):
        if self.doc:
            self.undo_stack.append((copy.deepcopy(self.doc.masks), copy.deepcopy(self.doc.boxes),
                                    copy.deepcopy(self.doc.points), self.doc.confirmed.copy(), self.doc.terminals_skipped))
            self.undo_stack = self.undo_stack[-15:]

    def undo(self):
        if self.task and self.task.isRunning():
            return
        if self.doc and self.undo_stack:
            self.doc.masks, self.doc.boxes, self.doc.points, self.doc.confirmed, self.doc.terminals_skipped = self.undo_stack.pop()
            self.dirty = True
            self.refresh()

    def changed(self):
        self.doc.changed(self.active)
        self.dirty = True
        self.refresh()

    def box_changed(self, box):
        if self.doc:
            self.snapshot()
            self.doc.boxes[self.active] = box
            self.doc.points[self.active] = []
            self.changed()

    def point_changed(self, point):
        if self.doc:
            self.snapshot()
            self.doc.points[self.active].append(point)
            self.changed()

    def clear_points(self):
        if self.doc:
            self.snapshot()
            self.doc.points[self.active] = []
            self.changed()

    def paint(self, start, end, erase):
        if not self.doc:
            return
        mask = self.doc.masks[self.active]
        value = 0 if erase else 1
        radius = max(1, self.canvas.brush_size//2)
        cv2.line(mask, start, end, value, self.canvas.brush_size)
        cv2.circle(mask, end, radius, value, -1)
        self.changed()

    def refresh(self):
        if self.doc:
            skipped = self.doc.terminals_skipped
            self.skip_terminals.blockSignals(True)
            self.skip_terminals.setChecked(skipped)
            self.skip_terminals.blockSignals(False)
            for index in (1, 2):
                self.target.model().item(index).setEnabled(not skipped)
            self.terminals_button.setEnabled(not skipped)
            if skipped and self.active != 'chip':
                self.target.setCurrentIndex(0)
            self.canvas.show_masks(self.doc, self.active)
            self.state.setText('\n'.join(f'{DISPLAY[k]}: {"スキップ（端子なし）" if skipped and k != "chip" else ("確定 ✓" if self.doc.confirmed[k] else "未確定")} / {int(self.doc.masks[k].sum())} px' for k in NAMES))
            self.setWindowTitle(f'Chip Annotator — {Path(self.doc.source).name}{" *" if self.dirty else ""}')

    def may_leave(self):
        if not self.dirty:
            return True
        choice = QMessageBox.question(self, '未保存の編集', '途中保存してから移動しますか？',
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
        if choice == QMessageBox.Save:
            return bool(self.save())
        return choice == QMessageBox.Discard

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, '画像', '', '画像 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)')
        if path:
            self.open_path(Path(path))

    def draft_path(self, path):
        return path.parent/'.chipannotator'/f'{document_key(path)}.chipproj'

    def open_path(self, path, check_leave=True):
        if check_leave and not self.may_leave():
            return
        def action():
            draft = self.draft_path(path)
            doc = load_project(draft) if draft.exists() else Document(read_image(path), str(path.resolve()))
            self.install(doc, draft if draft.exists() else None)
        self.safe(action)

    def install(self, doc, project_path=None):
        self.doc, self.project_path = doc, project_path
        self.undo_stack = []
        self.dirty = False
        self.target.setCurrentIndex(0)
        self.active = 'chip'
        self.canvas.set_image(doc.image)
        self.refresh()

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, '画像フォルダ（直下のみ）')
        if not folder or not self.may_leave():
            return
        self.folder = Path(folder)
        self.list.clear()
        for p in sorted(self.folder.iterdir()):
            if p.suffix.lower() in EXTENSIONS:
                self.list.addItem(p.name)
                self.list.item(self.list.count()-1).setData(Qt.UserRole, str(p))
        if self.list.count():
            self.list.setCurrentRow(0)
            self.open_path(Path(self.list.item(0).data(Qt.UserRole)), check_leave=False)

    def navigate(self, offset):
        row = self.list.currentRow()+offset
        if 0 <= row < self.list.count():
            self.open_path(Path(self.list.item(row).data(Qt.UserRole)))
            if self.doc and self.doc.source == str(Path(self.list.item(row).data(Qt.UserRole)).resolve()):
                self.list.setCurrentRow(row)

    def save(self, save_as=False):
        if not self.doc:
            return False
        path = self.project_path
        if not path or save_as:
            path, _ = QFileDialog.getSaveFileName(self, '途中保存（画像も含みます）',
                str(self.draft_path(Path(self.doc.source))), 'Chip project (*.chipproj)')
            if not path:
                return False
            if not str(path).lower().endswith('.chipproj'):
                path = str(path)+'.chipproj'
        def action():
            save_project(self.doc, path)
            self.project_path = Path(path)
            self.dirty = False
            self.refresh()
            self.statusBar().showMessage(f'保存済み: {path}')
            return True
        return self.safe(action)

    def autosave(self):
        if self.doc and self.dirty and not (self.task and self.task.isRunning()):
            # Recovery file stays separate from explicitly saved project.
            try:
                recovery = (self.project_path or self.draft_path(Path(self.doc.source))).with_suffix('.autosave.chipproj')
                save_project(self.doc, recovery)
                self.statusBar().showMessage(f'復旧用自動保存: {recovery}')
            except Exception as e:
                self.statusBar().showMessage(f'自動保存に失敗: {e} — 名前を付けて保存を利用してください')

    def resume(self):
        path, _ = QFileDialog.getOpenFileName(self, '途中保存から再開（自動保存も選択可能）', '', 'Chip project (*.chipproj)')
        if path and self.may_leave():
            self.safe(lambda: self.install(load_project(path), Path(path)))

    def get_backend(self):
        weights = self.settings.value('weights', str(ROOT/'models'/'mobile_sam.pt'))
        device = self.settings.value('device', 'auto')
        if self.backend is None or str(self.backend.weights) != weights or self.backend.requested_device != device:
            self.backend = MobileSAM(weights, device)
        return self.backend

    def run_task(self, fn, done):
        if self.task and self.task.isRunning():
            return
        self.cancel.clear()
        self.task = Task(fn)
        for widget in self.controls:
            widget.setEnabled(False)
        for action in self.file_actions:
            action.setEnabled(False)
        self.task.progress.connect(self.statusBar().showMessage)
        self.task.result.connect(lambda result: self.receive_result(done, result))
        self.task.failed.connect(self.receive_error)
        self.task.finished.connect(self.task_finished)
        self.statusBar().showMessage('処理中… 初回はモデル読込・画像特徴計算に時間がかかります')
        self.task.start()

    def receive_result(self, done, result):
        try:
            done(result)
            self.statusBar().showMessage('処理完了。結果を確認して、必要に応じて修正・確定してください。')
        except Exception as e:
            logging.exception('Result handling failed')
            self.receive_error(str(e))

    def receive_error(self, message):
        self.statusBar().showMessage('処理に失敗しました。エラー内容を確認してください。')
        self.error(message)

    def task_finished(self):
        for widget in self.controls:
            widget.setEnabled(True)
        for action in self.file_actions:
            action.setEnabled(True)
        if self.backend and self.backend.device:
            self.device_label.setText(f'SAM: {self.backend.device} / MobileSAM vit_t')
        self.refresh()

    def propose(self):
        if not self.doc:
            return
        name = self.active
        if name != 'chip' and not self.doc.confirmed['chip']:
            return self.error('先にチップを確定してください')
        if self.doc.masks[name].any():
            answer = QMessageBox.question(self, 'SAM再提案', '現在の対象マスクをSAMの結果に置き換えます。続けますか？（戻す操作が可能）')
            if answer != QMessageBox.Yes:
                return
        backend = self.get_backend()
        image, box, points = self.doc.image, self.doc.boxes.get(name), copy.deepcopy(self.doc.points[name])
        restrict = self.doc.masks['chip'].copy() if name != 'chip' else None
        def done(mask):
            self.snapshot()
            self.doc.masks[name] = mask
            self.doc.changed(name)
            self.dirty = True
            self.refresh()
        self.run_task(lambda _: backend.predict(image, box, points, restrict), done)

    def rectangle_mask(self):
        if not self.doc:
            return
        if self.active not in self.doc.boxes:
            return self.error('矩形ツールで対象を囲んでください')
        self.snapshot()
        mask = np.zeros(self.doc.image.shape[:2], np.uint8)
        x1, y1, x2, y2 = self.doc.boxes[self.active]
        mask[y1:y2+1, x1:x2+1] = 1
        self.doc.masks[self.active] = mask
        self.changed()

    def confirm(self):
        if not self.doc:
            return
        name = self.active
        if name != 'chip' and not self.doc.confirmed['chip']:
            return self.error('先にチップを確定してください')
        if not self.doc.masks[name].any():
            return self.error('マスクが空です')
        self.snapshot()
        self.doc.confirmed[name] = True
        self.dirty = True
        self.refresh()

    def terminals(self):
        if self.doc and self.doc.terminals_skipped:
            return self.error('端子なしを解除すると端子を編集できます')
        if not self.doc or not self.doc.confirmed['chip']:
            return self.error('チップマスクを修正して確定してください')
        if any(self.doc.masks[k].any() for k in ('left', 'right')):
            if QMessageBox.question(self, '左右を再提案', '左右の現在のマスク・点を置き換えます。続けますか？') != QMessageBox.Yes:
                return
        backend, image, chip = self.get_backend(), self.doc.image, self.doc.masks['chip'].copy()
        def work(progress):
            candidates = terminal_candidates(image, chip)
            result = {}
            for name, candidate in candidates.items():
                progress(f'{DISPLAY[name]}をSAMで抽出中…')
                box = bbox(candidate)
                # Pad candidate box but keep prompts within the source image.
                h, w = chip.shape
                box = [max(0, box[0]-3), max(0, box[1]-3), min(w-1, box[2]+3), min(h-1, box[3]+3)]
                distance = cv2.distanceTransform(candidate, cv2.DIST_L2, 3)
                y, x = np.unravel_index(distance.argmax(), distance.shape)
                points = [[int(x), int(y), 1]]
                result[name] = (backend.predict(image, box, points, chip), box, points)
            return result
        def done(result):
            self.snapshot()
            for name, (mask, box, points) in result.items():
                self.doc.masks[name], self.doc.boxes[name], self.doc.points[name] = mask, box, points
                self.doc.confirmed[name] = False
            self.dirty = True
            self.target.setCurrentIndex(1)
            self.refresh()
        self.run_task(work, done)

    def export(self):
        if not self.doc:
            return
        if not self.doc.ready_for_export:
            return self.error('チップと必要な端子を確定してください。端子なしの場合はスキップを選べます。')
        root = QFileDialog.getExistingDirectory(self, 'YOLOデータセット出力先', str(self.output_root or ''))
        if root:
            def action():
                key, report = export_yolo(self.doc, root)
                self.output_root = Path(root)
                warnings = [f'{DISPLAY[k]}: IoU={v["iou"]:.4f}, 穴={v["holes"]}, 領域={v["components"]}'
                            for k, v in report.items() if v['holes'] or v['components'] > 1 or v['iou'] < .995]
                QMessageBox.information(self, '書き出し完了', f'{root}\n{key}\n' + ('chip 1個（端子スキップ）\n' if self.doc.terminals_skipped else 'chip 1個 + terminal 2個\n') +
                    ('\nYOLO変換の注意（正確な形状はPNGマスクを参照）:\n'+'\n'.join(warnings) if warnings else ''))
            self.safe(action)

    def configure(self):
        dialog = QDialog(self)
        dialog.setWindowTitle('MobileSAM設定')
        form = QFormLayout(dialog)
        weights = QLineEdit(self.settings.value('weights', str(ROOT/'models'/'mobile_sam.pt')))
        select = QPushButton('重みファイルを選ぶ')
        def browse():
            path, _ = QFileDialog.getOpenFileName(dialog, '公式MobileSAM vit_t重み', '', 'PyTorch (*.pt *.pth)')
            if path:
                weights.setText(path)
        select.clicked.connect(browse)
        device = QComboBox()
        device.addItems(['auto', 'cuda', 'cpu'])
        device.setCurrentText(self.settings.value('device', 'auto'))
        form.addRow('モデル', weights)
        form.addRow(select)
        form.addRow('実行デバイス', device)
        form.addRow(QLabel('未導入時: setup_model.bat を実行。\nRTX 5070 Ti: CUDA 12.8対応PyTorchを使用。\nCUDA失敗時はエラーを表示。CPUへ明示切替できます。'))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() == QDialog.Accepted:
            self.settings.setValue('weights', weights.text())
            self.settings.setValue('device', device.currentText())
            self.backend = None
            self.device_label.setText('SAM: 未読込（次回推論で設定反映）')

    def batch(self):
        if not self.doc or not self.folder or 'chip' not in self.doc.boxes:
            return self.error('画像フォルダを開き、基準画像のチップ矩形を描いてください')
        if not self.may_leave():
            return
        if QMessageBox.question(self, '同じ配置の画像へ一括提案',
            '現在の矩形を画像サイズで正規化して全画像に適用します。\n同じ配置・向きの画像向けです。結果は未確定の途中保存です。\n既存プロジェクトはスキップします。各画像で確認・確定後に書き出してください。') != QMessageBox.Yes:
            return
        h, w = self.doc.image.shape[:2]
        normalized = np.array(self.doc.boxes['chip'])/[w, h, w, h]
        skip_terminals = self.doc.terminals_skipped
        paths = [Path(self.list.item(i).data(Qt.UserRole)) for i in range(self.list.count())]
        backend = self.get_backend()
        def work(progress):
            backend.load()  # Fail early, don't generate a folder of identical setup errors.
            report = []
            for index, path in enumerate(paths):
                if self.cancel.is_set():
                    report.append('停止しました。完了した途中保存は保持されています。')
                    break
                target = self.draft_path(path)
                if target.exists():
                    report.append(f'SKIP {path.name}')
                    continue
                progress(f'一括提案 {index+1}/{len(paths)}: {path.name}')
                try:
                    image = read_image(path)
                    ih, iw = image.shape[:2]
                    box = np.rint(normalized*[iw, ih, iw, ih]).astype(int)
                    box = np.clip(box, 0, [iw-1, ih-1, iw-1, ih-1]).tolist()
                    doc = Document(image, str(path.resolve()))
                    doc.terminals_skipped = skip_terminals
                    doc.boxes['chip'] = box
                    doc.masks['chip'] = backend.predict(image, box)
                    # Proposed chip is deliberately unconfirmed. Terminal estimates
                    # are also drafts and require inspection of the chip first.
                    try:
                        for name, candidate in ({} if skip_terminals else terminal_candidates(image, doc.masks['chip'])).items():
                            b = bbox(candidate)
                            distance = cv2.distanceTransform(candidate, cv2.DIST_L2, 3)
                            y, x = np.unravel_index(distance.argmax(), distance.shape)
                            pts = [[int(x), int(y), 1]]
                            doc.masks[name] = backend.predict(image, b, pts, doc.masks['chip'])
                            doc.boxes[name], doc.points[name] = b, pts
                    except Exception as e:
                        report.append(f'WARN {path.name}: 端子要手動指定: {e}')
                    save_project(doc, target)
                    report.append(f'OK {path.name}')
                except Exception as e:
                    report.append(f'ERROR {path.name}: {e}')
            log = self.folder/'.chipannotator'/'batch_report.txt'
            log.parent.mkdir(exist_ok=True)
            log.write_text('\n'.join(report), encoding='utf-8')
            return log
        self.run_task(work, lambda log: QMessageBox.information(self, '一括提案終了',
            f'結果とエラー一覧: {log}\n画像を開き直すと途中保存を読み込みます。全対象を確認・確定してください。'))

    def batch_export(self):
        if not self.may_leave():
            return
        source = QFileDialog.getExistingDirectory(self, '確定済みchipprojを保存したフォルダ',
            str(self.folder/'.chipannotator') if self.folder else '')
        if not source:
            return
        target = QFileDialog.getExistingDirectory(self, 'YOLO出力先')
        if not target:
            return
        paths = sorted(p for p in Path(source).glob('*.chipproj') if '.autosave.' not in p.name)
        def work(progress):
            report = []
            for i, path in enumerate(paths):
                if self.cancel.is_set():
                    report.append('停止しました')
                    break
                progress(f'一括書き出し {i+1}/{len(paths)}: {path.name}')
                try:
                    doc = load_project(path)
                    if not doc.ready_for_export:
                        report.append(f'SKIP 未確定: {path.name}')
                        continue
                    key, diagnostics = export_yolo(doc, target)
                    report.append(f'OK {path.name}: {key} {diagnostics}')
                except Exception as e:
                    report.append(f'ERROR {path.name}: {e}')
            log = Path(target)/'export_report.txt'
            log.write_text('\n'.join(report), encoding='utf-8')
            return log
        self.run_task(work, lambda log: QMessageBox.information(self, '一括書き出し終了', f'結果: {log}'))

    def closeEvent(self, event):
        if self.task and self.task.isRunning():
            self.cancel.set()
            self.statusBar().showMessage('処理中です。一括処理は現在の画像が完了すると停止します。終了後に閉じてください。')
            event.ignore()
        elif self.may_leave():
            event.accept()
        else:
            event.ignore()


def configure_application(app):
    # Qt's offscreen/minimal platform may not enumerate native Windows fonts.
    if not QFontDatabase.families():
        candidate = Path(os.environ.get('WINDIR', 'C:/Windows'))/'Fonts'/'YuGothR.ttc'
        if candidate.is_file():
            QFontDatabase.addApplicationFont(str(candidate))
    app.setFont(QFont('Yu Gothic UI', 10))
    app.setStyle('Fusion')


def main():
    import sys
    logging.basicConfig(filename=str(ROOT/'chip_annotator.log'), level=logging.INFO,
                        encoding='utf-8', format='%(asctime)s %(levelname)s %(message)s')
    app = QApplication(sys.argv)
    configure_application(app)
    window = Window()
    window.show()
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if path.suffix == '.chipproj':
            window.safe(lambda: window.install(load_project(path), path))
        else:
            window.open_path(path)
    return app.exec()

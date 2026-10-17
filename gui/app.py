"""FractalAutoencoder GUI. Run from the repository root:  python -m gui.app"""
import sys
import threading
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QPlainTextEdit, QPushButton,
                               QSpinBox, QTabWidget, QVBoxLayout, QWidget)

from gui import bridge  # noqa: E402


class Job(QThread):
    log = Signal(str)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn, self._stop = fn, threading.Event()

    def should_stop(self):
        return self._stop.is_set()

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            self.finished_ok.emit(self.fn(self))
        except Exception:
            self.failed.emit(traceback.format_exc())


def gray_pixmap(arr, width):
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape
    return QPixmap.fromImage(QImage(arr.data, w, h, w, QImage.Format_Grayscale8).copy()).scaledToWidth(width, Qt.SmoothTransformation)


def rgb_pixmap(rgb, width):
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()).scaledToWidth(width, Qt.SmoothTransformation)


class Pic(QLabel):
    def __init__(self, text="", size=300):
        super().__init__(text)
        self.setMinimumSize(size, size)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#111;color:#888")
        self.size_hint = size

    def gray(self, arr):
        self.setPixmap(gray_pixmap(arr, self.size_hint))

    def rgb(self, arr):
        self.setPixmap(rgb_pixmap(arr, max(min(self.width(), 900), 200)))


class LogBox(QPlainTextEdit):
    def __init__(self, height=90):
        super().__init__()
        self.setReadOnly(True)
        self.setMaximumBlockCount(300)
        self.setMaximumHeight(height)

    def add(self, text):
        self.appendPlainText(text)


class Runner:
    def __init__(self, log, buttons=()):
        self.log, self.buttons, self.job = log, buttons, None

    def start(self, fn, on_done=None):
        if self.job is not None and self.job.isRunning():
            return
        self.job = Job(fn)
        self.job.log.connect(self.log.add)
        self.job.failed.connect(lambda tb: self.log.add(tb.strip().splitlines()[-1]))
        if on_done:
            self.job.finished_ok.connect(on_done)
        self.job.finished.connect(lambda: [b.setEnabled(True) for b in self.buttons])
        for b in self.buttons:
            b.setEnabled(False)
        self.job.start()

    def stop(self):
        if self.job is not None:
            self.job.stop()


def row(*widgets):
    w = QWidget(); l = QHBoxLayout(w); l.setContentsMargins(0, 0, 0, 0)
    for x in widgets:
        l.addWidget(x)
    return w


class State:
    def __init__(self):
        self.loaded, self.folder, self.files = None, "", []


class ModelTab(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.model_path = QLineEdit(); b_model = QPushButton("Model (.pth)..."); b_model.clicked.connect(self.pick_model)
        self.folder = QLineEdit(); b_folder = QPushButton("Image folder..."); b_folder.clicked.connect(self.pick_folder)
        self.classes_path = QLineEdit(str(bridge.CLASSES_FILE)); b_cls = QPushButton("Class names..."); b_cls.clicked.connect(self.pick_classes)
        b_load = QPushButton("Load model"); b_load.clicked.connect(self.load)
        self.info = QLabel("No model loaded"); self.info.setWordWrap(True)
        self.files = QListWidget(); self.files.currentRowChanged.connect(lambda _: self.show_image())
        self.original, self.recon = Pic("original"), Pic("reconstruction")
        self.probs = QLabel(""); self.probs.setWordWrap(True)
        self.second = QLineEdit(); self.second.setReadOnly(True)
        b_second = QPushButton("Interpolate with the selected image..."); b_second.clicked.connect(self.interpolate)
        self.strip = QLabel(); self.log = LogBox(60)
        form = QFormLayout()
        form.addRow(b_model, self.model_path); form.addRow(b_folder, self.folder); form.addRow(b_cls, self.classes_path)
        form.addRow(b_load)
        left = QVBoxLayout(); left.addLayout(form); left.addWidget(self.info); left.addWidget(self.files, 1)
        pics = QHBoxLayout(); pics.addWidget(self.original); pics.addWidget(self.recon)
        right = QVBoxLayout(); right.addLayout(pics); right.addWidget(self.probs); right.addWidget(b_second); right.addWidget(self.strip)
        top = QHBoxLayout(); top.addLayout(left, 1); top.addLayout(right, 2)
        lay = QVBoxLayout(self); lay.addLayout(top); lay.addWidget(self.log)
        self.first_path = None

    def pick_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Checkpoint", "", "PyTorch (*.pth *.pt)")
        if path:
            self.model_path.setText(path)

    def pick_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Image folder")
        if path:
            self.folder.setText(path)
            self.state.folder = path
            self.state.files = bridge.list_images(path, limit=2000)
            self.files.clear()
            for f in self.state.files:
                self.files.addItem(QListWidgetItem(str(Path(f).relative_to(path))))
            self.log.add(f"{len(self.state.files)} images listed (first 2000)")

    def pick_classes(self):
        path, _ = QFileDialog.getOpenFileName(self, "Class names (one per line)", "", "Text (*.txt)")
        if path:
            self.classes_path.setText(path)

    def load(self):
        try:
            classes = [l.strip() for l in Path(self.classes_path.text()).read_text().splitlines() if l.strip()]
            self.state.loaded = bridge.Loaded(self.model_path.text(), classes)
            self.info.setText(self.state.loaded.describe())
            self.show_image()
        except Exception as exc:
            self.info.setText(f"Could not load: {exc}")

    def current_file(self):
        row_ = self.files.currentRow()
        return self.state.files[row_] if 0 <= row_ < len(self.state.files) else None

    def show_image(self):
        path, loaded = self.current_file(), self.state.loaded
        if path is None or loaded is None:
            return
        o, r, probs, mu, logvar, mse = bridge.reconstruct(loaded, path)
        self.original.gray(o); self.recon.gray(r)
        text = f"reconstruction error (MSE) {mse:.4f}"
        if probs is not None:
            top = np.argsort(probs)[::-1][:3]
            text += "\nclasses: " + ", ".join(f"{loaded.classes[i] if i < len(loaded.classes) else i} {probs[i]:.2f}" for i in top)
        self.probs.setText(text)
        self.first_path = path

    def interpolate(self):
        if self.first_path is None or self.state.loaded is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Second image", self.state.folder, "Images (*.png *.jpg *.jpeg)")
        if path:
            frames = bridge.interpolate(self.state.loaded, self.first_path, path, 7)
            strip = np.concatenate([f[::4, ::4] for f in frames], axis=1)
            self.strip.setPixmap(gray_pixmap(strip, 900))


class MapTab(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state, self.latents, self.files, self.emb, self.labels = state, None, [], None, None
        self.count = QSpinBox(); self.count.setRange(10, 5000); self.count.setValue(300)
        self.embed = QComboBox(); self.embed.addItems(["pca", "tsne"] + (["umap"] if bridge.umap_available() else []))
        self.method = QComboBox(); self.method.addItems(["kmeans", "dbscan", "lof"])
        self.k = QSpinBox(); self.k.setRange(2, 50); self.k.setValue(8)
        self.eps = QDoubleSpinBox(); self.eps.setRange(0.01, 100); self.eps.setValue(5.0)
        self.contam = QDoubleSpinBox(); self.contam.setRange(0.01, 0.5); self.contam.setValue(0.1)
        self.b_encode = QPushButton("Encode images"); self.b_encode.clicked.connect(self.encode)
        self.b_plot = QPushButton("Cluster and plot"); self.b_plot.clicked.connect(self.plot)
        b_export = QPushButton("Copy clusters into folders..."); b_export.clicked.connect(self.export)
        self.view = Pic("encode the images, then plot", 400)
        self.sample = QListWidget(); self.sample.currentRowChanged.connect(self.neighbours)
        self.neigh = QLabel(); self.neigh.setWordWrap(True)
        self.log = LogBox(80)
        self.runner = Runner(self.log, (self.b_encode,))
        form = QFormLayout()
        form.addRow("Images to encode (random sample)", self.count)
        form.addRow("2-D projection", self.embed); form.addRow("Grouping", self.method)
        form.addRow("k-means clusters", self.k); form.addRow("DBSCAN eps (standardised units)", self.eps)
        form.addRow("Outlier fraction (LOF)", self.contam)
        left = QVBoxLayout(); left.addLayout(form); left.addWidget(self.b_encode); left.addWidget(self.b_plot); left.addWidget(b_export)
        left.addWidget(QLabel("Images (select one to see its nearest neighbours)")); left.addWidget(self.sample, 1); left.addWidget(self.neigh)
        top = QHBoxLayout(); top.addLayout(left, 1); top.addWidget(self.view, 3)
        lay = QVBoxLayout(self); lay.addLayout(top); lay.addWidget(self.log)

    def encode(self):
        loaded, folder = self.state.loaded, self.state.folder
        if loaded is None or not folder:
            self.log.add("load a model and choose an image folder first"); return
        n = self.count.value()

        def work(job):
            files = bridge.list_images(folder, limit=n, shuffle=True)
            z, lv = bridge.encode_folder(loaded, files, 16, job, job.log.emit)
            return files, z

        def done(result):
            self.files, self.latents = result
            self.sample.clear()
            for f in self.files:
                self.sample.addItem(Path(f).name)
            self.log.add(f"{len(self.files)} images encoded")
        self.runner.start(work, done)

    def plot(self):
        if self.latents is None or len(self.latents) < 5:
            self.log.add("encode at least 5 images first"); return
        try:
            self.emb = bridge.embed(self.latents, self.embed.currentText())
            labels, scores = bridge.cluster(self.latents, self.method.currentText(), self.k.value(), self.eps.value(),
                                            contamination=self.contam.value())
        except Exception as exc:
            self.log.add(f"{type(exc).__name__}: {exc}"); return
        self.labels = labels
        self.view.rgb(bridge.scatter_picture(self.emb, labels))
        found = sorted(set(int(v) for v in labels))
        self.log.add(f"groups: {found}  sizes: {[int((labels == v).sum()) for v in found]}")

    def export(self):
        if self.labels is None:
            return
        out = QFileDialog.getExistingDirectory(self, "Output folder (files are copied, nothing is deleted)")
        if out:
            self.log.add(f"copied {bridge.export_clusters(self.files, self.labels, out)} files")

    def neighbours(self, row):
        if self.latents is None or row < 0:
            return
        near = bridge.nearest(self.latents, row, 6)
        self.neigh.setText("nearest in the latent space:\n" + "\n".join(Path(self.files[i]).name for i in near))


class TrainTab(QWidget):
    def __init__(self):
        super().__init__()
        self.kind = QComboBox()
        for key, label in bridge.KINDS.items():
            self.kind.addItem(label, key)
        self.data = QLineEdit(); b_data = QPushButton("Dataset folder (ImageFolder with an 'unlabeled' class)..."); b_data.clicked.connect(self.pick_data)
        self.out = QLineEdit("trained.pth"); b_out = QPushButton("Save as..."); b_out.clicked.connect(self.pick_out)
        self.resume = QLineEdit(); b_res = QPushButton("Resume from..."); b_res.clicked.connect(self.pick_resume)
        self.latent = QSpinBox(); self.latent.setRange(8, 2048); self.latent.setValue(256)
        self.epochs = QSpinBox(); self.epochs.setRange(1, 100000); self.epochs.setValue(20)
        self.batch = QSpinBox(); self.batch.setRange(1, 128); self.batch.setValue(8)
        self.warm = QSpinBox(); self.warm.setRange(0, 1000); self.warm.setValue(10)
        self.lr = QDoubleSpinBox(); self.lr.setDecimals(6); self.lr.setRange(1e-7, 1); self.lr.setValue(1e-4); self.lr.setSingleStep(1e-5)
        self.beta = QDoubleSpinBox(); self.beta.setRange(0, 10); self.beta.setValue(0.1)
        self.alpha = QDoubleSpinBox(); self.alpha.setRange(0, 1e7); self.alpha.setValue(30000)
        self.scaler = QCheckBox("Legacy: balance the two losses with the running-mean GradientScaler instead of alpha")
        self.aug = QCheckBox("Repeat labelled images 5 times with random flips and inversions (original behaviour)"); self.aug.setChecked(True)
        self.b_run = QPushButton("Train"); b_stop = QPushButton("Stop")
        self.b_run.clicked.connect(self.run); b_stop.clicked.connect(lambda: self.runner.stop())
        self.log = LogBox(260)
        self.runner = Runner(self.log, (self.b_run,))
        form = QFormLayout()
        form.addRow("Model", self.kind); form.addRow(b_data, self.data); form.addRow(b_out, self.out); form.addRow(b_res, self.resume)
        form.addRow("Latent dimension", self.latent); form.addRow("Epochs", self.epochs); form.addRow("Batch size", self.batch)
        form.addRow("Warm-up epochs (classifier frozen)", self.warm); form.addRow("Learning rate", self.lr)
        form.addRow("KL weight beta", self.beta); form.addRow("Classifier loss weight alpha", self.alpha)
        form.addRow(self.scaler); form.addRow(self.aug)
        lay = QVBoxLayout(self); lay.addLayout(form); lay.addWidget(row(self.b_run, b_stop)); lay.addWidget(self.log, 1)

    def pick_data(self):
        p = QFileDialog.getExistingDirectory(self, "Dataset")
        if p:
            self.data.setText(p)

    def pick_out(self):
        p, _ = QFileDialog.getSaveFileName(self, "Checkpoint", self.out.text(), "PyTorch (*.pth)")
        if p:
            self.out.setText(p)

    def pick_resume(self):
        p, _ = QFileDialog.getOpenFileName(self, "Checkpoint", "", "PyTorch (*.pth)")
        if p:
            self.resume.setText(p)

    def run(self):
        if not self.data.text():
            self.log.add("choose a dataset folder"); return
        args = dict(kind=self.kind.currentData(), data_folder=self.data.text(), out_path=self.out.text(), latent=self.latent.value(),
                    epochs=self.epochs.value(), batch=self.batch.value(), lr=self.lr.value(), beta=self.beta.value(),
                    alpha=self.alpha.value(), warmup_epochs=self.warm.value(), use_gradient_scaler=self.scaler.isChecked(),
                    augment=self.aug.isChecked(), resume=self.resume.text() or None)
        self.runner.start(lambda job: bridge.train(job=job, log=job.log.emit, **args))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FractalAutoencoder")
        state = State()
        tabs = QTabWidget()
        for w, name in ((ModelTab(state), "Model and images"), (MapTab(state), "Latent map"), (TrainTab(), "Train")):
            tabs.addTab(w, name)
        self.setCentralWidget(tabs)
        self.resize(1250, 800)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

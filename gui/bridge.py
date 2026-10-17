"""Glue between the PySide6 GUI and the FractalAutoencoder models.

The original scripts import each other as ``src.X`` and pick folders from the machine's host name (``Configuration``), run
``Trainer`` as soon as the module is imported, and ``LatentSpaceAnalyzer.splitintofoldersbykmeans`` deletes a folder with
``shutil.rmtree`` before it copies files. The GUI keeps the model classes and the dataset class, and re-implements the
analysis and the training loop with explicit paths, so that it never deletes anything and never writes into the source tree.
"""
import copy
import os
import random
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
AE_DIR = REPO / "autoencoder"
CLASSES_FILE = REPO / "classes.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_EXT = (".png", ".jpg", ".jpeg")


def _alias():
    # The sources import each other as ``src.X``.
    if "src" not in sys.modules:
        pkg = types.ModuleType("src")
        pkg.__path__ = [str(AE_DIR)]
        sys.modules["src"] = pkg
    if "src.Configuration" not in sys.modules:
        # Configuration chooses folders from the host name; the GUI passes paths explicitly instead.
        stub = types.ModuleType("src.Configuration")

        class Configuration:
            def __init__(self):
                self.device = DEVICE
                self.datasetfolder = self.modelspath = self.classificationfolder = ""
                self.num_epochs, self.modelname = 1000000, ""
        stub.Configuration = Configuration
        sys.modules["src.Configuration"] = stub


_alias()

from src.ConvAutoEncoder import ConvAutoencoder  # noqa: E402
from src.ConvAutoEncoderwithClassifier import ConvAutoencoderwithClassifier  # noqa: E402
from src.ConvVAE import ConvVAE  # noqa: E402
from src.ConvVAEwithClassifier import ConvVAEwithClassifier  # noqa: E402
from src.DatasetLoader import ImageFolderwithFiles  # noqa: E402
from src.GradientScaler import GradientScaler  # noqa: E402

KINDS = {
    "vae_cls": "Convolutional VAE with a classifier (the latest model)",
    "vae": "Convolutional VAE, no classifier (legacy)",
    "ae_cls": "Plain autoencoder with a classifier (legacy)",
    "ae": "Plain autoencoder (legacy)",
}


def default_classes():
    return [line.strip() for line in CLASSES_FILE.read_text().splitlines() if line.strip()]


def detect(state):
    """Infer the architecture of a checkpoint from its tensor names and shapes."""
    if "fc_mu.weight" in state:
        latent = state["fc_mu.weight"].shape[0]
        if "classifier.4.weight" in state:
            return "vae_cls", latent, state["classifier.4.weight"].shape[0]
        return "vae", latent, 0
    if "encoder.13.weight" in state:
        latent = state["encoder.13.weight"].shape[0]
        if "classifier.2.weight" in state:
            return "ae_cls", latent, state["classifier.2.weight"].shape[0]
        return "ae", latent, 0
    raise ValueError("unknown checkpoint layout")


def build(kind, latent, classes=None, class_count=0, beta=0.1):
    names = list(classes or [])
    if class_count and len(names) != class_count:
        names = [f"class {i}" for i in range(class_count)]
    if kind == "vae_cls":
        return ConvVAEwithClassifier(names, latent_dim=latent, beta=beta)
    if kind == "vae":
        return ConvVAE(latent_dim=latent, beta=beta)
    if kind == "ae_cls":
        return ConvAutoencoderwithClassifier(names, latent_dim=latent)
    return ConvAutoencoder(latent_dim=latent)


class Loaded:
    """A checkpoint plus uniform encode / decode / classify calls."""

    def __init__(self, path, classes=None):
        state = torch.load(path, map_location="cpu")
        self.kind, self.latent, self.class_count = detect(state)
        self.model = build(self.kind, self.latent, classes or default_classes(), self.class_count)
        self._fit_classifier(state)
        self.model.load_state_dict(state)
        self.model.to(DEVICE).eval()
        self.path = str(path)
        self.classes = getattr(self.model, "classes", [])

    def _fit_classifier(self, state):
        """Older checkpoints used a different classifier depth (e.g. 256-256-256-22); rebuild the head to match."""
        keys = sorted((int(k.split(".")[1]) for k in state if k.startswith("classifier.") and k.endswith(".weight")))
        if not keys:
            return
        layers, activation = [], (nn.LeakyReLU(0.2) if self.kind == "vae_cls" else nn.ReLU())
        for idx in range(keys[-1] + 1):
            if idx in keys:
                w = state[f"classifier.{idx}.weight"]
                layers.append(nn.Linear(w.shape[1], w.shape[0]))
            else:
                layers.append(copy.deepcopy(activation))
        self.model.classifier = nn.Sequential(*layers)

    @property
    def is_vae(self):
        return self.kind.startswith("vae")

    @torch.no_grad()
    def encode(self, x):
        """x (n, 1, 512, 512) -> latent codes (n, latent) and, for VAEs, log-variances."""
        x = x.to(DEVICE)
        if self.is_vae:
            mu, logvar = self.model.encode(x)
            return mu, logvar
        return self.model.encoder(x), None

    @torch.no_grad()
    def decode(self, z):
        z = z.to(DEVICE)
        if self.is_vae:
            return self.model.decode(z)
        return self.model.decoder(z)

    @torch.no_grad()
    def classify(self, z):
        z = z.to(DEVICE)
        if self.kind in ("vae_cls", "ae_cls"):
            return F.softmax(self.model.classifier(z), dim=1)
        return None

    def describe(self):
        n = sum(p.numel() for p in self.model.parameters())
        return f"{KINDS[self.kind]}\nlatent dimension {self.latent}, {self.class_count} classes, {n / 1e6:.1f} M parameters, on {DEVICE}"


def load_image(path):
    img = Image.open(path).convert("L").resize((512, 512), Image.BILINEAR)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)[None, None]


def list_images(folder, limit=None, shuffle=False, seed=0):
    files = []
    for root, _, names in os.walk(folder):
        for name in names:
            if name.lower().endswith(IMAGE_EXT):
                files.append(os.path.join(root, name))
        if limit and not shuffle and len(files) >= limit:
            break
    if shuffle:
        random.Random(seed).shuffle(files)
    return files[:limit] if limit else files


def to_uint8(t):
    return (t.detach().cpu().squeeze().clamp(0, 1).numpy() * 255).astype(np.uint8)


def reconstruct(loaded, path):
    """Returns original, reconstruction (uint8 images), per-class probabilities (or None), mu and logvar vectors."""
    x = load_image(path)
    mu, logvar = loaded.encode(x)
    recon = loaded.decode(mu)
    probs = loaded.classify(mu)
    mse = float(F.mse_loss(recon.cpu(), x))
    return to_uint8(x), to_uint8(recon), (probs[0].cpu().numpy() if probs is not None else None), mu[0].cpu().numpy(), \
        (logvar[0].cpu().numpy() if logvar is not None else None), mse


def interpolate(loaded, path_a, path_b, steps=7):
    ma, _ = loaded.encode(load_image(path_a))
    mb, _ = loaded.encode(load_image(path_b))
    frames = []
    for t in np.linspace(0, 1, steps):
        frames.append(to_uint8(loaded.decode(ma * (1 - t) + mb * t)))
    return frames


def encode_folder(loaded, files, batch=16, job=None, log=print):
    codes, logvars = [], []
    for i in range(0, len(files), batch):
        if job is not None and job.should_stop():
            break
        x = torch.cat([load_image(p) for p in files[i:i + batch]])
        mu, logvar = loaded.encode(x)
        codes.append(mu.cpu().numpy())
        if logvar is not None:
            logvars.append(logvar.cpu().numpy())
        log(f"encoded {min(i + batch, len(files))}/{len(files)}")
    z = np.vstack(codes) if codes else np.zeros((0, loaded.latent))
    lv = np.vstack(logvars) if logvars else None
    return z[:len(files)], (lv[:len(files)] if lv is not None else None)


# ---------------------------------------------------------------- latent analysis (from LatentSpaceAnalyzer)
def embed(latents, method="pca", seed=42):
    if method == "umap":
        import umap
        return umap.UMAP(random_state=seed).fit_transform(latents)
    if method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, random_state=seed, perplexity=min(30, max(2, len(latents) // 4))).fit_transform(latents)
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=seed).fit_transform(latents)


def umap_available():
    try:
        import umap  # noqa: F401
        return True
    except Exception:
        return False


def cluster(latents, method="kmeans", k=8, eps=0.5, min_samples=5, contamination=0.1):
    """K-means (original default k=2), DBSCAN, or Local Outlier Factor on standardised codes."""
    from sklearn.preprocessing import StandardScaler
    scaled = StandardScaler().fit_transform(latents)
    if method == "kmeans":
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled), None
    if method == "dbscan":
        from sklearn.cluster import DBSCAN
        return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(scaled), None
    from sklearn.neighbors import LocalOutlierFactor
    lof = LocalOutlierFactor(contamination=contamination, novelty=True).fit(scaled)
    return lof.predict(scaled), -lof.score_samples(scaled)


def scatter_picture(emb, labels, width=860, height=620):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    plt.scatter(emb[:, 0], emb[:, 1], c=labels, s=14, cmap="tab20")
    plt.grid(True, alpha=.3)
    plt.tight_layout()
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return rgb


def export_clusters(files, labels, out_dir):
    """Copy (never move or delete) the files into out_dir/<label>/ ."""
    for f, lab in zip(files, labels):
        target = Path(out_dir) / str(int(lab))
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, target / Path(f).name)
    return len(files)


def nearest(latents, index, count=8):
    d = np.linalg.norm(latents - latents[index], axis=1)
    return list(np.argsort(d)[1:count + 1])


# ---------------------------------------------------------------- training (from Trainer.py)
def train(kind, data_folder, out_path, latent=256, epochs=10, batch=8, lr=1e-4, beta=0.1, alpha=30000.0,
          warmup_epochs=10, use_gradient_scaler=False, augment=True, workers=0, job=None, log=print, on_epoch=None, resume=None):
    """Two-phase training of the original: the classifier is frozen for `warmup_epochs` epochs and only the
    reconstruction (VAE) loss is used; afterwards labelled images add `alpha` x a multi-label BCE loss.

    ``data_folder`` is an ImageFolder directory that must contain an ``unlabeled`` class folder, as in the original.
    Plain models (legacy) are trained with MSE on every image.
    """
    from torchvision import transforms
    from torch.utils.data import DataLoader
    tf = transforms.Compose([transforms.Grayscale(), transforms.Resize((512, 512)), transforms.ToTensor()])

    def valid(path):
        return path.lower().endswith(IMAGE_EXT)
    dataset = ImageFolderwithFiles(root=data_folder, transform=tf, is_valid_file=valid, allow_empty=True,
                                   class_multipliers=5 if augment else 1)
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=workers, drop_last=False)
    classes = dataset.classes
    model = build(kind, latent, classes, len(classes), beta)
    if resume:
        model.load_state_dict(torch.load(resume, map_location="cpu"))
    model.to(DEVICE).train()
    has_classifier = kind in ("vae_cls", "ae_cls")
    is_vae = kind.startswith("vae")
    scaler = GradientScaler(device=DEVICE) if use_gradient_scaler else None

    def set_classifier(flag):
        if has_classifier:
            for p in model.classifier.parameters():
                p.requires_grad = flag
    set_classifier(False)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    history = []
    for epoch in range(epochs):
        if job is not None and job.should_stop():
            break
        training_classifier = has_classifier and epoch >= warmup_epochs
        if training_classifier and epoch == warmup_epochs:
            set_classifier(True)
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
        running, count = 0.0, 0
        for i, (images, target, _, labeled) in enumerate(loader):
            if job is not None and job.should_stop():
                break
            images, target, labeled = images.to(DEVICE), target.to(DEVICE), labeled.to(DEVICE)
            recon_total = torch.zeros(1, device=DEVICE)
            class_loss = torch.zeros(1, device=DEVICE)
            if is_vae:
                unlabeled = images[~labeled]
                if unlabeled.shape[0]:
                    out = model(unlabeled)
                    total, _, _ = model.loss_function(out[0], unlabeled, out[1], out[2])
                    recon_total = recon_total + total
                if training_classifier or not has_classifier:
                    labeled_images = images[labeled]
                    if labeled_images.shape[0]:
                        out = model(labeled_images)
                        total, _, _ = model.loss_function(out[0], labeled_images, out[1], out[2])
                        recon_total = recon_total + total
                        if has_classifier:
                            class_loss = F.binary_cross_entropy_with_logits(
                                out[3], F.one_hot(target[labeled], num_classes=len(classes)).float())
            else:
                out = model(images)
                recon_total = F.mse_loss(out[0], images)
                if has_classifier and training_classifier and labeled.any():
                    class_loss = F.binary_cross_entropy_with_logits(
                        out[2][labeled], F.one_hot(target[labeled], num_classes=len(classes)).float())
            if scaler is not None:
                a, b = scaler.update(recon_total.sum(), class_loss.sum())
                loss = a + b
            else:
                loss = recon_total + class_loss * (alpha if is_vae else 1.0)
            if not loss.requires_grad:   # a batch with only labelled images during the warm-up phase
                continue
            optimizer.zero_grad()
            loss.sum().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += float(loss.sum()); count += 1
            if (i + 1) % 10 == 0:
                log(f"epoch {epoch + 1} batch {i + 1}/{len(loader)} loss {float(loss.sum()):.3f}")
        mean = running / max(count, 1)
        history.append(mean)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_path)
        log(f"epoch {epoch + 1}/{epochs}: mean loss {mean:.4f}, saved {out_path}")
        if on_epoch:
            on_epoch(epoch + 1, mean)
    return history

# FractalAutoencoder

Convolutional (variational) autoencoders on 512 x 512 fractal pictures. The 64-256 numbers they produce for each picture are used to **group** the pictures, to find
**outliers**, and, with a classifier head trained on a few dozen hand-labelled pictures, to **name** the family of a fractal.

- How it works, what worked and what did not: [docs/PROJECT.md](docs/PROJECT.md)
- Features that were tried and dropped: [docs/UNUSED_FEATURES.md](docs/UNUSED_FEATURES.md)
- `classes.txt`: the 32 class folders of the last dataset (the order the classifier uses)

## GUI

```
pip install -r requirements-gui.txt
python -m gui.app
```

- **Model and images**: open any checkpoint (the architecture is detected), browse an image folder, see the reconstruction, the top classes and interpolate between two images.
- **Latent map**: encode a sample, project to 2-D, group (k-means, DBSCAN, outliers) and copy the groups into folders.
- **Train**: train any of the four models on an `ImageFolder` dataset that has an `unlabeled` class.

Weights are not included (150-300 MB each); train one with the Train tab. Checkpoints are Python pickles: only open files you created.

## Original scripts

`autoencoder/` (imports `src.X`, so it must be importable as `src`; `Trainer.py` starts training when it is run).

## License

[PolyForm Noncommercial 1.0.0](LICENSE). Personal, research and hobby use are allowed; commercial use is not.

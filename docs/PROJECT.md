# FractalAutoencoder: how it works, what worked, what did not

Written from the code, the seven checkpoints in `model/` (October-November 2024) and the folders left by the clustering runs. Statements marked *(inferred)*
are conclusions from that evidence, not recorded results.

## The idea

Fractal pictures come out of the generators by the thousand, and there is no cheap way to ask "is this a good one?". A **convolutional autoencoder** gives each 512 x 512
grey image a short code (64 to 256 numbers). Nearby codes mean similar pictures, so the code space can be used to **group** pictures, to **find outliers** and to
**classify** them into named families, which can in turn be used as a fitness term in a generator (RewriteFractal's `Sieve` calls it).

## The models

| Model | File | What it is |
|---|---|---|
| `ConvAutoencoder` | `ConvAutoEncoder.py` | 6 strided convolutions (1 -> 1024 channels, 512 -> 8 pixels), one linear layer to the code, mirror decoder, ReLU, sigmoid output. MSE. |
| `ConvAutoencoderwithClassifier` | `ConvAutoEncoderwithClassifier.py` | The same plus a two-layer classifier on the code. 32 classes in `autoencodervae2.pth`. |
| `ConvVAE` | `ConvVAE.py` | Same layout with batch norm and LeakyReLU, mean and log-variance heads (code 64, 128 or 256), beta-VAE loss (BCE + beta x KL, beta = 0.1), small initial weights on the heads. |
| `ConvVAEwithClassifier` | `ConvVAEwithClassifier.py` | VAE with a three-layer classifier on the sampled code; the latest model (256-dim code). |

Checkpoints in `model/`: `autoencoder.pth` (plain, 128), `autoencodervae.pth` and `autoencodervae2ld128.pth` (VAE, 128), `autoencodervae2ld64.pth` (VAE, 64),
`autoencodervae2.pth` (plain autoencoder with a 32-class classifier, 128), `autoencodervae2withclass.pth` (VAE with a 256-256-256-22 classifier, 256) and `autoencodervae2back.pth`.
The GUI reads the architecture from the tensor names and shapes, so any of them can be opened.

## Training (`Trainer.py`)

1. Images come from folders of the generator's output (`graphfractals/GraphFractals/lowres/`). The folder name is the class. One folder is called `unlabeled` and holds almost everything:
   **37,875 files in `unlabeled` against 1 to 57 images in each of the 31 named classes** (Floral 1, Hyperboid 3, mandala 4, ...). So this is semi-supervised.
2. Labelled images are repeated 5 times, and on each access get one of five random changes (invert the greys, flip, transpose, or a combination).
3. For the first 10 epochs the classifier is frozen and only the VAE loss on unlabelled images is used. After that, labelled images are added and a multi-label BCE classification loss with weight **30000** is summed with the VAE loss.
4. Adam at 1e-4, gradient clipping at 1.0, up to a million epochs (it was stopped by hand). Every 2 epochs the model is saved and a figure of originals, reconstructions and codes is drawn. Every 10 epochs the dataset is reloaded so that newly sorted images are used.

## Latent space analysis (`LatentSpaceAnalyser.py`)

Encode all images, standardise the codes, then K-means (the number of clusters was fixed to 2 in the code, the silhouette search is commented out), DBSCAN, or Local Outlier Factor, and a UMAP picture with file names. `splitintofoldersbykmeans` copies each image into `classificationfolder/kmeans/<cluster>/`
(it first **deletes** that folder). The folder left on disk has 20 clusters of 84 to 3,658 images (sorted by the 64-dim VAE) *(inferred: a run with more clusters than the code's current default)*.

## What worked

- **A VAE code is good enough to separate families.** The sorted folders show recognisable groups, and 31 named classes (Dentelle, Floral, mandala, spheroid, ...) came from looking at the groups and naming them.
- **Semi-supervision with a handful of labels.** Repeating and augmenting 1-57 labelled images per class, with the classifier frozen at first, was enough for the classifier to name the top three classes in the reconstruction figure.
- **The beta-VAE version trained more stably than the plain one**, which is why the later checkpoints are all VAEs: batch normalisation, small head initialisation, clamped reconstruction and a NaN check were all added.
- **Code sizes 64 / 128 / 256** were all tried; 64 was used for the clustering run and 256 for the classifier.
- **Use in a generator.** RewriteFractal's `Sieve` loaded `autoencodervae2.pth` and scored candidates by the probability of wanted and unwanted classes.

## What did not work

- **The GPU memory and speed.** Every image is 512 x 512 and the first linear layer has 65,536 inputs, so each checkpoint is 140-290 MB and training with batches of 40 on an 8-12 GB card was the limit.
- **Loss balancing.** A running-mean `GradientScaler` was written to normalise the two losses and then replaced by a hard-coded weight of 30000, which was tuned by hand *(inferred)*.
- **A classifier head that changed shape** between checkpoints (2, 3 and 4 linear layers, 22 then 32 classes), so most checkpoints cannot be loaded by the final `Trainer` without adjustment.
- **K-means with 2 clusters.** The silhouette search for the best number of clusters is commented out, and the shipped default is 2.
- **Batches with only labelled images** raise "element 0 of tensors does not require grad" during the frozen phase (the loss is a constant). The original `Trainer` only avoids this because unlabelled images dominate.
- **`Inference.py`** (in this folder) never loads the weights and indexes the wrong axis; the working version lives in RewriteFractal.
- **Hard-coded paths and host names** in `Configuration.py` (`arch-laptop`, `archdesktop`), with `src.X` imports that stop working once the folder is not called `src`.
- **`initialisationvisualisation.py`** is empty.

## Lessons

1. Keep the unlabeled data and the labelled data in the same loader but with different treatment (repeat and augment the rare labels).
2. Freeze the classifier first, then train jointly: the code gets a shape before it is asked to separate classes.
3. A small code (64) is enough to cluster; a larger one (256) helps the classifier.
4. Save the architecture with the checkpoint (a config next to it), or the weights cannot be reopened a month later.
5. Copy, never delete, when sorting files by cluster.

## Running the original code

Put `autoencoder/` where the imports find it as `src` (e.g. rename it to `src`) and set the folders in `Configuration.py`. Training starts when `Trainer.py` is run.

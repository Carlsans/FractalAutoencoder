# Features that exist in the code but are not part of the final workflow

| # | Feature | Where in the code | Why unused | In the GUI |
|---|---|---|---|---|
| 1 | Plain autoencoder | `ConvAutoEncoder.py`, MSE loss in `Trainer` | Replaced by the VAE | Train > Model "Plain autoencoder (legacy)" |
| 2 | Plain autoencoder with classifier | `ConvAutoEncoderwithClassifier.py` (from RewriteFractal) | Used by RewriteFractal's sieve only | Train, and the Model tab opens its checkpoint (`autoencodervae2.pth`) |
| 3 | VAE without classifier | `ConvVAE.py` | Superseded by the classifier version | Train > "Convolutional VAE, no classifier (legacy)" |
| 4 | GradientScaler | `GradientScaler.py` | Commented out in `Trainer` in favour of weight 30000 | Train checkbox |
| 5 | K-means, DBSCAN and outlier detection | `LatentSpaceAnalyser.cluster_*`, `detect_outliers` | Called from commented lines | Latent map > Grouping (k-means with a chosen k, DBSCAN, LOF) |
| 6 | UMAP picture | `LatentSpaceAnalyser.visualize_latent_space` | Needs `umap-learn` | Latent map > projection (UMAP when installed; PCA and t-SNE otherwise) |
| 7 | Copying images into cluster folders | `splitintofoldersbykmeans` | Deletes the target folder first | Latent map > "Copy clusters into folders" (copies only) |
| 8 | Reconstruction figure | `Trainer.visualize_reconstruction` | Needs an open matplotlib window | Model and images tab: original, reconstruction, top classes |
| 9 | Latent interpolation | none (new) | Natural use of the decoder that was never exposed | Model and images tab |
| 10 | Dataset reload every 10 epochs | `Trainer` | Only useful when sorting images by hand while training | Not integrated: the GUI trains on a fixed folder |
| 11 | Silhouette search for k | commented in `cluster_kmeans` | Slow | Not integrated: choose k by eye on the map |
| 12 | `initialisationvisualisation.py` | empty file | Never written | Not integrated |

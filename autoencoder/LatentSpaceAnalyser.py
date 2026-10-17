import time

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.neighbors import LocalOutlierFactor
import umap

from src.Configuration import Configuration
from src.ConvVAE import ConvVAE
from src.DatasetLoader import DatasetLoader
from pathlib import Path
import shutil
import os
import glob

class LatentSpaceAnalyzer:
    def __init__(self, autoencoder, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.conf = Configuration()
        self.autoencoder = autoencoder
        self.device = device
        self.autoencoder.to(device)
        self.autoencoder.eval()

    def extract_latent_vectors(self, dataloader):
        """Extract latent vectors from the dataset using the encoder."""
        latent_vectors = []
        filesname = []
        with torch.no_grad():
            for batch in dataloader:
                filesname.append(batch[2])
                if isinstance(batch, (tuple, list)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(self.device)
                if len(x.shape) == 3:
                    x = torch.unsqueeze(x,dim=0)
                #print("Input shape ",x.shape)
                # Assuming the autoencoder has an encode method or we can access the encoder
                latent = self.autoencoder.encode(x)
                #print("latent",latent)
                latent_vectors.append(np.hstack([latent[0].cpu().numpy(),latent[1].cpu().numpy()]))
        return np.vstack(latent_vectors),filesname

    def cluster_kmeans(self, latent_vectors,filesname, n_clusters_range=range(2, 50)):
        """Perform K-means clustering with optimal number of clusters."""
        # Scale the features
        scaler = StandardScaler()
        scaled_vectors = scaler.fit_transform(latent_vectors)

        # Find optimal number of clusters using silhouette score
        # best_score = -1
        # best_n_clusters = 2
        #
        # for n_clusters in n_clusters_range:
        #     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        #     labels = kmeans.fit_predict(scaled_vectors)
        #     score = silhouette_score(scaled_vectors, labels)
        #
        #     if score > best_score:
        #         best_score = score
        #         best_n_clusters = n_clusters

        # Fit final model with optimal number of clusters
        # N_cluster overidder
        final_kmeans = KMeans(n_clusters=2, random_state=42)
        labels = final_kmeans.fit_predict(scaled_vectors)
        if not filesname is None:
            [self.appendtologfile("kmeancluster"+str(labels[i]),filesname[i]) for i in range(len(labels))]
        return labels, final_kmeans
    def splitintofoldersbykmeans(self):
        datasetloader = DatasetLoader()
        datasetloader.loadDataset()
        latent_vectors,filesname = self.extract_latent_vectors(datasetloader.dataloader)
        filesnamesinglelist = []
        for i in filesname:
            for j in i:
                filesnamesinglelist.append(j)
        filesname = filesnamesinglelist
        print("filesname", filesname)
        filesonly = [ i[i.rfind('/')+1:] for i in filesname]
        kmeans,_= self.cluster_kmeans(latent_vectors,None)
        kmeans = list(kmeans)
        print("Kmeans :",list(kmeans))
        print("Files :",filesonly)

        # Delete previous files
        kmeansdir = self.conf.classificationfolder+"kmeans/"
        if os.path.exists(kmeansdir) and os.path.isdir(kmeansdir):
            shutil.rmtree(kmeansdir)
        # Copy files
        for i in range(len(kmeans)):
            folderpath = kmeansdir +str(kmeans[i])
            Path(folderpath).mkdir(parents=True, exist_ok=True)
            print("Copying ",filesname[i],"to",folderpath+ "/" +filesonly[i])
            shutil.copyfile(filesname[i], folderpath+ "/" +filesonly[i])


    def detect_outliers(self, latent_vectors,filesname, contamination=0.1):
        """Detect outliers using Local Outlier Factor."""
        scaler = StandardScaler()
        scaled_vectors = scaler.fit_transform(latent_vectors)

        lof = LocalOutlierFactor(contamination=contamination, novelty=True)
        lof.fit(scaled_vectors)

        # Get outlier scores (-1 for outliers, 1 for inliers)
        outlier_labels = lof.predict(scaled_vectors)
        # Get outlier scores (negative scores indicate outliers)
        outlier_scores = -lof.score_samples(scaled_vectors)
        [self.appendtologfile("detect_outliers label: "+str(outlier_labels[i])+" score: " + str(outlier_scores[i]),filesname[i]) for i in range(len(outlier_labels))]

        return outlier_labels, outlier_scores

    def cluster_dbscan(self, latent_vectors,filesname, eps=0.5, min_samples=5):
        """Perform DBSCAN clustering which inherently identifies outliers."""
        scaler = StandardScaler()
        scaled_vectors = scaler.fit_transform(latent_vectors)

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(scaled_vectors)
        [self.appendtologfile("dbscan label: "+str(labels[i]),filesname[i]) for i in range(len(labels))]
        return labels

    def visualize_latent_space(self, latent_vectors,filesname,savewithtimestamp=False):
        """Reduce dimensionality to 2D for visualization using UMAP."""
        reducer = umap.UMAP(random_state=42)
        embedding = reducer.fit_transform(latent_vectors)
        print(embedding.shape)
        if not len(filesname[0]) == 1:
            filesname = filesname[0]
        filesonly = [ i[i.rfind('/')+1:-4] for i in filesname]
        labels, _ = self.cluster_kmeans(latent_vectors, None)
        self.plot_embedding_with_labels(embedding,labels,filesonly,savewithtimestamp)


        return embedding

    def plot_embedding_with_labels(self, embedding,labels, filename =None,savewithtimestamp=False):
        """
        Plots a 2D embedding with optional labels for each point.

        Parameters:
            embedding (numpy.ndarray): A (x, 2) shaped numpy array with 2D coordinates.
            labels (list or array, optional): A list or array of labels for each point. Defaults to index if None.
        """

        x_coords, y_coords = embedding[:, 0], embedding[:, 1]

        plt.figure(figsize=(10, 8))
        plt.scatter(x_coords,y_coords,c=labels)
        if filename is None:
            filename = range(len(embedding))

        for i, label in enumerate(filename):
            plt.text(x_coords[i], y_coords[i], str(label), fontsize=4, ha='right')

        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("2D Embedding Visualization with Labels")
        plt.grid(True)
        timestamp = ""
        if savewithtimestamp:
            timestamp = time.strftime("%d-%H%M%S")
        plt.savefig(self.conf.datasetfolder  + '/embeddings2d'+ timestamp +'.jpg')

        if savewithtimestamp:
            plt.close()
        else:
            plt.show()
    def appendtologfile(self,property,imagefilepath):
        print(property,imagefilepath)
        path = imagefilepath[0:imagefilepath.rfind('/')]+'/'
        imagename = imagefilepath[imagefilepath.rfind('/') + 1:imagefilepath.rfind('.')]
        print(path ,imagename)
        logfilepathandfilename = path + imagename + ".log"
        with open(logfilepathandfilename, 'a+') as f:
            f.write(property +'\n')


def analyze_dataset(autoencoder, dataloader):
    """Convenience function to perform complete analysis."""
    analyzer = LatentSpaceAnalyzer(autoencoder)

    # Extract latent vectors
    latent_vectors,filesname = analyzer.extract_latent_vectors(dataloader)

    # Perform clustering
    #kmeans_labels, kmeans_model = analyzer.cluster_kmeans(latent_vectors,filesname )
    #dbscan_labels = analyzer.cluster_dbscan(latent_vectors,filesname )

    # Detect outliers
    #outlier_labels, outlier_scores = analyzer.detect_outliers(latent_vectors,filesname )

    # Create 2D visualization
    embedding = analyzer.visualize_latent_space(latent_vectors,filesname )

def splitintofoldersbykmean():
    conf = Configuration()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #net = torch.load(conf.modelspath+conf.modelname,map_location=torch.device(device))
    net = torch.load("/home/carl/PycharmProjects/FractalAutoencoder/model/autoencodervae2ld64.pth",map_location=torch.device(device))

    model = ConvVAE(latent_dim=64)
    model.load_state_dict(net)
    LSanalyser = LatentSpaceAnalyzer(model)
    LSanalyser.splitintofoldersbykmeans()
    # datasetloader = DatasetLoader()
    # datasetloader.loadDataset()
    # print(analyze_dataset(model,datasetloader.dataset))

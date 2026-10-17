from src.Configuration import Configuration
from src.ConvVAEwithClassifier import ConvVAEwithClassifier
from src.DatasetLoader import DatasetLoader


class Inference:
    def __init__(self):
        self.conf = Configuration()
        self.device = self.conf.device
        self.modelspath = self.conf.modelspath
        self.modelname = self.conf.modelname

        self.datasetloader = DatasetLoader()
        self.model = ConvVAEwithClassifier(self.datasetloader.dataset.classes, latent_dim=256)

    def getClassProbability(self,imagetensor,classname):
        classidx = self.datasetloader.dataset.class_to_idx[classname]
        reconstructed, mu, logvar, classes = self.model(imagetensor)
        return classes[classidx]
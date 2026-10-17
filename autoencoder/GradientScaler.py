import torch


class GradientScaler:
    def __init__(self, alpha=0.99,device = 'cpu'):
        self.alpha = alpha
        self.device = device
        self.running_mean1 = None
        self.running_mean2 = None

    def update(self, loss1, loss2):
        # Get current loss values
        with torch.no_grad():
            if self.running_mean1 is None:
                if not loss1 is None:
                    self.running_mean1 = loss1.item()

                if not loss2 is None:
                    self.running_mean2 = loss2.item()

            else:
                if not loss1 is None:
                    self.running_mean1 = self.alpha * self.running_mean1 + (1 - self.alpha) * loss1.item()
                else:
                    self.running_mean1 = torch.zeros(1).to(self.device)
                if not loss2 is None:
                    self.running_mean2 = self.alpha * self.running_mean2 + (1 - self.alpha) * loss2.item()
                else:
                    self.running_mean2 = torch.zeros(1).to(self.device)

        # Compute scaling factors
        if not loss1 is None:
            scale1 = 1.0 / (self.running_mean1 + 1e-8)
        else:
            scale1 = 0
            loss1 = torch.zeros(1).to(self.device)

        if not loss2 is None:
            scale2 = 1.0 / (self.running_mean2 + 1e-8)
        else:
            scale2 = 0
            loss2 = torch.zeros(1).to(self.device)

        return scale1 * loss1 , scale2 * loss2
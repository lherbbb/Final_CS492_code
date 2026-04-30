import torch
import torch.nn.functional as F


class DecodeClassifier(torch.nn.Module):
    def __init__(self, classifier, vae, scaling_factor, size=(224, 224)):
        super().__init__()
        self.classifier = classifier
        self.vae = vae
        self.scaling_factor = scaling_factor
        self.size = size

    def forward(self, x, ):
        out = self.vae.decode(x / self.scaling_factor)["sample"]
        out = out / 2 + 0.5
        out = F.interpolate(out, size=self.size, mode="bilinear")
        out = self.classifier(out)
        return out

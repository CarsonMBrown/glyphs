import torch
from torchvision.models import AlexNet


class AlexNetSimilarity(AlexNet):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        print(x.shape())
        x = self.avgpool(x)
        print(x.shape())
        x = torch.flatten(x, 1)
        print(x.shape())
        x = self.classifier(x)
        print(x.shape())
        return x

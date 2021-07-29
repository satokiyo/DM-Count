import torch
from . import initialization as init


class ClassificationModel(torch.nn.Module):

    def initialize(self):
        init.initialize_head(self.classification_head)
        if self.classification_head_aux is not None:
            init.initialize_head(self.classification_head_aux)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)

        labels = self.classification_head(features[-1])

        if self.classification_head_aux is not None:
            labels_aux = self.classification_head(features[-1])
            return labels, labels_aux

        return labels

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x

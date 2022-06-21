import torch
import torch.nn as nn
import torch.nn.functional as F
from ucdr.models import FastSCNN, Classifer

__all__ = ["FastSCNNNoisyDecoders", "FastSCNNJointSegAuxDecoderModel"]


class FastSCNNDropOutDecoder(nn.Module):
    def __init__(self, drop_rate=0.3, num_classes=40, decoder=None):
        super().__init__()
        self.dropout = nn.Dropout2d(p=drop_rate)
        self.decoder = Classifer(128, num_classes) if decoder is None else decoder

    def forward(self, x, size=(640, 320), *ig, **ign):
        x = self.dropout(x)
        x = self.decoder(x)  # BS,40,48,48
        x = F.interpolate(x, size, mode="bilinear", align_corners=True)
        return x


class FastSCNNNoisyDecoders(nn.Module):
    def __init__(self, n_decoders, dropout, num_classes=40):
        super().__init__()
        self.decoders = nn.ModuleList(
            [FastSCNNDropOutDecoder(drop_rate=dropout, num_classes=num_classes) for _ in range(n_decoders)]
        )

    def forward(self, x):
        return [decoder(x) for decoder in self.decoders]

    def optim_parameters(self, args):
        return [{"params": self.parameters(), "lr": 10 * args.learning_rate}]


class FastSCNNJointSegAuxDecoderModel(nn.Module):
    def __init__(self, fastscnn, auxmodule):
        super().__init__()
        self.fastscnn = fastscnn
        self.aux_decoders = auxmodule

    def forward(self, x, training=False):
        seg_pred, features = self.fastscnn(x)
        if not training:
            return seg_pred
        perturbed_out = self.aux_decoders(features)
        input_size = x.size()[2:]
        perturbed_out = [self.interp(x, input_size) for x in perturbed_out]

        return seg_pred, perturbed_out

    def optim_parameters(self, args):
        def get_1x_lr_params_NOscale(fastscnn):
            """
            This generator returns all the parameters of the net except for
            the last classification layer. Note that for each batchnorm layer,
            requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
            any batchnorm parameter
            """
            b = [
                fastscnn._md["learn_to_down"].parameters(),
                fastscnn._md["extractor"].parameters(),
                fastscnn._md["fusion"].parameters(),
            ]

            for j in range(len(b)):
                for i in b[j]:
                    yield i

        def get_10x_lr_params(fastscnn):
            """
            This generator returns all the parameters for the last layer of the net,
            which does the classification of pixel into classes
            """
            b = [fastscnn._md["classifier"].parameters()]
            for j in range(len(b)):
                for i in b[j]:
                    yield i

        def optim_parameters(fastscnn, args):
            return [
                {"params": get_1x_lr_params_NOscale(fastscnn), "lr": args.learning_rate},
                {"params": get_10x_lr_params(fastscnn), "lr": 10 * args.learning_rate},
            ]

        return optim_parameters(self.fastscnn, args) + self.aux_decoders.optim_parameters(args)

    def interp(self, x1, input_size):
        return F.interpolate(x1, size=input_size, mode="bilinear", align_corners=True)

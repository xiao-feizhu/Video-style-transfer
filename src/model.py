import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# --------- AdaIN core ----------

def calc_mean_std(feat: torch.Tensor, eps: float = 1e-5):
    """
    Calculate channel-wise mean and std of feature maps.
    feat: (N, C, H, W)
    """
    N, C = feat.size()[:2]
    feat_var = feat.view(N, C, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adain(content_feat: torch.Tensor,
          style_feat: torch.Tensor,
          eps: float = 1e-5) -> torch.Tensor:
    """
    Adaptive Instance Normalization.
    content_feat, style_feat: (N, C, H, W)
    """
    c_mean, c_std = calc_mean_std(content_feat, eps)
    s_mean, s_std = calc_mean_std(style_feat, eps)
    normalized = (content_feat - c_mean) / c_std
    return normalized * s_std + s_mean


# --------- VGG encoder ----------

# class VGGEncoder(nn.Module):
#     """
#     VGG19 encoder truncated at relu4_1.
#     We also expose intermediate features for style loss.
#     """
#     def __init__(self):
#         super().__init__()
#         vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

#         # Up to relu4_1 (index 21 is relu4_1, but we slice until 21 inclusive)
#         # indices: relu1_1:1, relu2_1:6, relu3_1:11, relu4_1:20
#         self.enc = nn.Sequential(*[vgg[i] for i in range(21)])

#         for p in self.parameters():
#             p.requires_grad = False

#         self.layer_ids = {
#             "relu1_1": 1,
#             "relu2_1": 6,
#             "relu3_1": 11,
#             "relu4_1": 20,
#         }

#     def forward(self, x):
#         """
#         Return list of features: [relu1_1, relu2_1, relu3_1, relu4_1]
#         """
#         feats = {}
#         for i, layer in enumerate(self.enc):
#             x = layer(x)
#             for name, idx in self.layer_ids.items():
#                 if i == idx:
#                     feats[name] = x

#         return [feats["relu1_1"], feats["relu2_1"],
#                 feats["relu3_1"], feats["relu4_1"]]

class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        # Slice EXACTLY as AdaIN official
        self.slice1 = nn.Sequential(*[vgg[i] for i in range(2)])    # relu1_1
        self.slice2 = nn.Sequential(*[vgg[i] for i in range(2, 7)])  # relu2_1
        self.slice3 = nn.Sequential(*[vgg[i] for i in range(7, 12)]) # relu3_1
        self.slice4 = nn.Sequential(*[vgg[i] for i in range(12, 21)]) # relu4_1  ← note: ends BEFORE 21

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h1 = h
        h = self.slice2(h)
        h2 = h
        h = self.slice3(h)
        h3 = h
        h = self.slice4(h)
        h4 = h
        return [h1, h2, h3, h4]

# --------- Decoder ----------

class Decoder(nn.Module):
    """
    Decoder network that inverts VGG features back to RGB image.
    Architecture inspired by original AdaIN implementation.
    Input: feature map at relu4_1 level (C=512).
    """
    def __init__(self):
        super().__init__()

        layers = []
        # relu4_1 feature: (512, H/16, W/16)

        def conv_relu(in_ch, out_ch):
            return nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, kernel_size=3),
                nn.ReLU(inplace=True),
            )

        # A sequence roughly mirroring VGG19 but upside-down
        layers += [conv_relu(512, 512)]
        layers += [nn.Upsample(scale_factor=2, mode="nearest")]  # 1/8
        layers += [conv_relu(512, 512), conv_relu(512, 512), conv_relu(512, 256)]
        layers += [nn.Upsample(scale_factor=2, mode="nearest")]  # 1/4
        layers += [conv_relu(256, 256), conv_relu(256, 256), conv_relu(256, 128)]
        layers += [nn.Upsample(scale_factor=2, mode="nearest")]  # 1/2
        layers += [conv_relu(128, 128), conv_relu(128, 64)]
        # layers += [nn.Upsample(scale_factor=2, mode="nearest")]  # 1
        # final conv (no ReLU)
        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, kernel_size=3)
        ]

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


# --------- Full AdaIN style transfer module ----------

class AdaINStyleTransfer(nn.Module):
    """
    Full AdaIN style transfer network:
    encoder (fixed VGG) + AdaIN + decoder (trainable).
    """
    def __init__(self, encoder: VGGEncoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        return self.encoder(x)

    def decode(self, t_features):
        return self.decoder(t_features)

    def forward(self, content, style, alpha: float = 1.0):
        """
        content, style: normalized for VGG input (ImageNet mean/std)
        alpha: style strength (0~1).
        """
        c_feats = self.encoder(content)
        s_feats = self.encoder(style)

        c4 = c_feats[-1]
        s4 = s_feats[-1]

        t = adain(c4, s4)
        t = alpha * t + (1.0 - alpha) * c4

        out = self.decoder(t)
        return out, c_feats, s_feats

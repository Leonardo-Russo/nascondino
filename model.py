import os
import shutil
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageFile
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import cv2
import csv
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from matplotlib.patches import ConnectionPatch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import CocoDetection
import argparse
import math
import torch.optim as optim

from torchvision.datasets import CIFAR10
from torchvision.datasets import VOCSegmentation
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# updated version from https://github.com/gngdb/pytorch-pca/blob/main/pca.py


class PCA_toTest(nn.Module):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components

    @staticmethod
    def _svd_flip(u, v, u_based_decision=True):
        """
        Adjusts the signs of the singular vectors from the SVD decomposition for
        deterministic output.

        This method ensures that the output remains consistent across different
        runs.

        Args:
            u (torch.Tensor): Left singular vectors tensor.
            v (torch.Tensor): Right singular vectors tensor.
            u_based_decision (bool, optional): If True, uses the left singular
              vectors to determine the sign flipping. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Adjusted left and right singular
              vectors tensors.
        """
        if u_based_decision:
            max_abs_cols = torch.argmax(torch.abs(u), dim=0)
            signs = torch.sign(u[max_abs_cols, torch.arange(u.shape[1])])
        else:
            max_abs_rows = torch.argmax(torch.abs(v), dim=1)
            signs = torch.sign(v[torch.arange(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, None]
        return u, v

    @torch.no_grad()
    def fit(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X)
        n, d = X.shape
        if self.n_components is not None:
            d = min(self.n_components, d)
        mean = X.mean(0)
        self.register_buffer("mean_", mean.view(1, -1))  # Keep mean as a row vector
        Z = X - self.mean_  # center
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Vt = Vh
        U, Vt = self._svd_flip(U, Vt)
        self.register_buffer("components_", Vt[:d])
        return self

    def forward(self, X):
        return self.transform(X)

    def transform(self, X):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(X - self.mean_, self.components_.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(Y, self.components_) + self.mean_


class PrivacyHead(nn.Module):
    def __init__(self, image_size=224, in_channels=768, out_size=1):
        super(PrivacyHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        self.image_size = image_size

    def forward(self, x_tokens_grid, cls_tokens, debug=False):
        if debug:
            print("shape 0: ", x_tokens_grid.shape)
        x_tokens_grid = self.relu(self.conv1(x_tokens_grid))
        if debug:
            print("shape 1: ", x_tokens_grid.shape)
        x_tokens_grid = F.interpolate(x_tokens_grid, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        if debug:
            print("shape 2: ", x_tokens_grid.shape)
        x_tokens_grid = self.relu(self.conv2(x_tokens_grid))
        if debug:
            print("shape 3: ", x_tokens_grid.shape)
        privacy_maps = torch.sigmoid(self.conv3(x_tokens_grid))

        return privacy_maps


class PrivacyDINO(nn.Module):
    def __init__(self, repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14", pretrained=True, pca_components=768, num_classes=10):
        super(PrivacyDINO, self).__init__()

        self.original_model = torch.hub.load(repo_name, model_name)

        self.patch_size = self.original_model.patch_size
        self.interpolate_offset = self.original_model.interpolate_offset
        self.interpolate_antialias = self.original_model.interpolate_antialias
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.original_model.to(self.device)

        self.patch_embed = self.original_model.patch_embed
        self.blocks = self.original_model.blocks
        self.norm = self.original_model.norm
        self.head = self.original_model.head
        self.cls_token = self.original_model.cls_token.clone()
        self.pos_embed = self.original_model.pos_embed.clone()

        # PCA compression
        self.pca_components = pca_components
        self.pca = PCA(n_components=pca_components)

        # Classification head for CIFAR-10
        self.classification_head = nn.Linear(pca_components, num_classes)

        # Freeze parameters if pretrained
        if pretrained:
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            for param in self.blocks.parameters():
                param.requires_grad = False
            for param in self.norm.parameters():
                param.requires_grad = False
            for param in self.head.parameters():
                param.requires_grad = False


    def forward(self, x, debug=False):
        # --- Pretrained DINOv2 Processing --- #
        self.original_model.eval()
        x_imgs = x

        x_dino = self.prepare_tokens(x_imgs)
        for blk in self.blocks:
            x_dino = blk(x_dino)
        x_dino = self.norm(x_dino)

        x_cls = x_dino[:, :1, :]  # CLS token
        x_tokens = x_dino[:, 1:, :]

        if debug:
            print("img shape: ", x_imgs.shape)
            print("tokens shape: ", x_tokens.shape)
            print("cls shape: ", x_cls.shape)

        # Apply PCA to reduce dimensionality of tokens
        batch_size, n_tokens, n_depth = x_tokens.shape
        x_tokens_reshaped = x_tokens.reshape(-1, n_depth).cpu().detach().numpy()

        if not hasattr(self, 'pca_fitted') or not self.pca_fitted:
            self.pca.fit(x_tokens_reshaped)
            self.pca_fitted = True

        x_tokens_compressed = self.pca.transform(x_tokens_reshaped)
        x_tokens_compressed = torch.tensor(x_tokens_compressed).view(batch_size, n_tokens, self.pca_components).to(self.device)

        # Apply PCA to CLS token
        x_cls_reshaped = x_cls.view(batch_size, -1).cpu().detach().numpy()
        x_cls_compressed = self.pca.transform(x_cls_reshaped)
        x_cls_compressed = torch.tensor(x_cls_compressed).view(batch_size, -1).to(self.device)

        # Use the CLS token for classification
        logits = self.classification_head(x_cls_compressed)

        return x_tokens_compressed, x_cls_compressed, logits



    def interpolate_pos_encoding(self, x, w, h):

        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)


    def prepare_tokens(self, x, debug=False):

        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.to(self.device).expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h).to(self.device)

        return x
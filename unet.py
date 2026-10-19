import pathlib
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pims
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pad
from tqdm import tqdm


class UNet(nn.Module):
    """
    The U-Net model. This is pretty much exactly the model from `Ronneberger et
    al. 2015 <http://arxiv.org/abs/1505.04597>`_ but the convolutions are padded
    to have the output size be equal to the input size. Downsampling is done
    using a learned maxpooling and upsampling using a learned transpose
    convolution with stride equal to 2.

    Parameters
    ----------
    in_channels : int
        The number of colour channels of the image.
    mid_channels : int
        The number of channels of the first convolution layer.
    out_channels : int
        The number of classes to classify.
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        dropout: bool = False,
        dropout_p: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels

        self.C01 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
        )
        self.C02 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.Dropout2d(p=dropout_p) if dropout else nn.Identity(),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
        )
        self.M1 = nn.MaxPool2d(2, 2)
        self.C11 = nn.Sequential(
            nn.Conv2d(mid_channels, 2 * mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * mid_channels),
            nn.ReLU(),
        )
        self.C12 = nn.Sequential(
            nn.Conv2d(2 * mid_channels, 2 * mid_channels, kernel_size=3, padding=1),
            nn.Dropout2d(p=dropout_p) if dropout else nn.Identity(),
            nn.BatchNorm2d(2 * mid_channels),
            nn.ReLU(),
        )
        self.M2 = nn.MaxPool2d(2, 2)
        self.C21 = nn.Sequential(
            nn.Conv2d(2 * mid_channels, 4 * mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(4 * mid_channels),
            nn.ReLU(),
        )
        self.C22 = nn.Sequential(
            nn.Conv2d(4 * mid_channels, 4 * mid_channels, kernel_size=3, padding=1),
            nn.Dropout2d(p=dropout_p) if dropout else nn.Identity(),
            nn.BatchNorm2d(4 * mid_channels),
            nn.ReLU(),
        )
        self.M3 = nn.MaxPool2d(2, 2)
        self.C31 = nn.Sequential(
            nn.Conv2d(4 * mid_channels, 8 * mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(8 * mid_channels),
            nn.ReLU(),
        )
        self.C32 = nn.Sequential(
            nn.Conv2d(8 * mid_channels, 8 * mid_channels, kernel_size=3, padding=1),
            nn.Dropout2d(p=dropout_p) if dropout else nn.Identity(),
            nn.BatchNorm2d(8 * mid_channels),
            nn.ReLU(),
        )
        self.M4 = nn.MaxPool2d(2, 2)

        self.C41 = nn.Sequential(
            nn.Conv2d(8 * mid_channels, 16 * mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(16 * mid_channels),
            nn.ReLU(),
        )
        self.C42 = nn.Sequential(
            nn.Conv2d(16 * mid_channels, 16 * mid_channels, kernel_size=3, padding=1),
            # nn.Dropout2d(p=dropout_p) if dropout else nn.Identity(),
            nn.BatchNorm2d(16 * mid_channels),
            nn.ReLU(),
        )

        self.CT1 = nn.ConvTranspose2d(
            16 * mid_channels, 8 * mid_channels, kernel_size=2, stride=2
        )
        self.C51 = nn.Sequential(
            nn.Conv2d(16 * mid_channels, 8 * mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(8 * mid_channels),
            nn.ReLU(),
        )
        self.C52 = nn.Sequential(
            nn.Conv2d(8 * mid_channels, 8 * mid_channels, kernel_size=3, padding=1),
            nn.Dropout2d(p=dropout_p) if dropout else nn.Identity(),
            nn.BatchNorm2d(8 * mid_channels),
            nn.ReLU(),
        )
        self.CT2 = nn.ConvTranspose2d(
            8 * mid_channels, 4 * mid_channels, kernel_size=2, stride=2
        )
        self.C61 = nn.Sequential(
            nn.Conv2d(8 * mid_channels, 4 * mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(4 * mid_channels),
            nn.ReLU(),
        )
        self.C62 = nn.Sequential(
            nn.Conv2d(4 * mid_channels, 4 * mid_channels, kernel_size=3, padding=1),
            nn.Dropout2d(p=dropout_p) if dropout else nn.Identity(),
            nn.BatchNorm2d(4 * mid_channels),
            nn.ReLU(),
        )
        self.CT3 = nn.ConvTranspose2d(
            4 * mid_channels, 2 * mid_channels, kernel_size=2, stride=2
        )
        self.C71 = nn.Sequential(
            nn.Conv2d(4 * mid_channels, 2 * mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * mid_channels),
            nn.ReLU(),
        )
        self.C72 = nn.Sequential(
            nn.Conv2d(2 * mid_channels, 2 * mid_channels, kernel_size=3, padding=1),
            nn.Dropout2d(p=dropout_p) if dropout else nn.Identity(),
            nn.BatchNorm2d(2 * mid_channels),
            nn.ReLU(),
        )
        self.CT4 = nn.ConvTranspose2d(
            2 * mid_channels, mid_channels, kernel_size=2, stride=2
        )
        self.C81 = nn.Sequential(
            nn.Conv2d(2 * mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
        )
        self.C82 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.Dropout2d(p=dropout_p) if dropout else nn.Identity(),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
        )
        self.C83 = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        The class method which defines a forward pass through the U-Net.

        Parameters
        ----------
        inp : torch.tensor
            The input to the network.

        Returns
        -------
        C83 : torch.tensor
            The output of the network.
        """
        C01 = self.C01(inp)
        C02 = self.C02(C01)
        M1 = self.M1(C02)
        C11 = self.C11(M1)
        C12 = self.C12(C11)
        M2 = self.M2(C12)
        C21 = self.C21(M2)
        C22 = self.C22(C21)
        M3 = self.M3(C22)
        C31 = self.C31(M3)
        C32 = self.C32(C31)
        M4 = self.M4(C32)

        C41 = self.C41(M4)
        C42 = self.C42(C41)

        CT1 = self.CT1(C42)
        C51 = self.C51(torch.cat([CT1, C32], dim=1))
        C52 = self.C52(C51)
        CT2 = self.CT2(C52)
        C61 = self.C61(torch.cat([CT2, C22], dim=1))
        C62 = self.C62(C61)
        CT3 = self.CT3(C62)
        C71 = self.C71(torch.cat([CT3, C12], dim=1))
        C72 = self.C72(C71)
        CT4 = self.CT4(C72)
        C81 = self.C81(torch.cat([CT4, C02], dim=1))
        C82 = self.C82(C81)
        C83 = self.C83(C82)

        return C83


def dice_loss(input: torch.tensor, target: torch.tensor) -> torch.tensor:
    """
    A function for the the Sørensen-Dice coefficient loss.

    Parameters
    ----------
    input : torch.tensor
        The generated class labels.
    target : torch.tensor
        The ground truth class labels.

    Returns
    -------
     : torch.tensor
        The dice coefficient loss.
    """
    smooth = 1.0

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


def weighted_mc_xentropy(
    input: torch.tensor,
    target: torch.tensor,
    log_var: torch.tensor,
    nb_mc: int,
    weight: torch.tensor,
    device: str,
    lambda_reg: float = 0.01,
) -> torch.tensor:
    mu_mc = torch.repeat(input, (nb_mc, 1, 1, 1)).to(device)
    std = torch.exp(log_var)
    noise = torch.randn(mu_mc.size(), device=device) * std
    prd = mu_mc + noise

    target = torch.repeat(target, (nb_mc, 1, 1, 1)).to(device)
    mc_x = nn.functional.binary_cross_entropy_with_logits(
        prd, target, weight=weight, reduction="none"
    )
    mc_x_mean = mc_x.view(nb_mc, -1).mean()

    log_var_penalty = torch.exp(-log_var).mean()

    return mc_x_mean + lambda_reg * log_var_penalty


def closest_number(n, m):
    q = int(n / m)
    n1 = q * m
    n2 = m * (q + 1)

    if n <= n1:
        return n1

    return n2


class UNetInference:
    """
    The class for doing inference with the U-Net.

    Parameters
    ----------
    unet_model : torch.nn.Module
        The U-Net model.
    batch_size : int
        The batch size to use for inference.
    device_id : torch.device or str or int, Optional
        The device to perform the inference on. Default is 0, so will use device `"cuda:0"`.
    """

    def __init__(
        self,
        unet_model: UNet,
        batch_size: int,
        device_id: Union[torch.device, str, int] = 0,
        n_mc: int = 20,
    ):
        self.unet = unet_model
        self.unet.eval()
        for m in self.unet.modules():
            if isinstance(m, torch.nn.Dropout2d):
                m.train()
        self.batch_size = batch_size
        self.n_mc = n_mc

        if isinstance(device_id, str):
            self.device = torch.device(device_id)
        elif isinstance(device_id, int):
            self.device = torch.device(f"cuda:{device_id}")
        else:
            self.device = device_id
        self.unet.to(self.device)

    def load_checkpoint(self, filename: Union[pathlib.Path, str]) -> None:
        """
        This class method loads a checkpoint for the model.

        Parameters
        ----------
        filename : str
            The path to the model to load.
        """

        if filename.is_file():
            print(f"=> loading checkpoint at {str(filename)}")
            chkpt = torch.load(filename, map_location=self.device)
            self.current_epoch = chkpt["epoch"]
            self.train_losses = chkpt["train_losses"]
            self.val_losses = chkpt["val_losses"]
            self.unet.load_state_dict(chkpt["model_state_dict"])

            train_l = self.train_losses[-1]
            val_l = self.val_losses[-1]
            print(
                f"=> loaded checkpoint at {str(filename)} at epoch {self.current_epoch} with training loss {train_l} and validation loss {val_l}."
            )
        else:
            print(f"=> no checkpoint found at {str(filename)}")

    def castle_valet(self, video: torch.tensor) -> torch.tensor:
        """
        The class method to perform the inference.

        Parameters
        ----------
        video : torch.tensor
            The video to segment.

        Returns
        -------
        preds : torch.tensor
            The U-Net masks.
        uncerts : torch.tensor
            The log variance of the U-Net logits.
        """
        preds = torch.zeros(video.shape[:-1], dtype=torch.uint8, device=self.device)
        uncerts = torch.zeros(video.shape[:-1], dtype=torch.float, device=self.device)
        video = torch.permute(video, (0, 3, 1, 2))
        y, x = video.shape[-2], video.shape[-1]
        y_close = closest_number(y, 16)
        x_close = closest_number(x, 16)
        y_pad = abs(y_close - y) // 2 if y != y_close else 0
        x_pad = abs(x_close - x) // 2 if x != x_close else 0
        dl = DataLoader(video, batch_size=self.batch_size, shuffle=False)

        for j, batch in enumerate(tqdm(dl)):
            # with torch.autocast(device_type="cuda"):
            with torch.no_grad():
                logits, stds = [], []
                batch = pad(batch, [x_pad, y_pad]).float().to(self.device)
                for _ in range(self.n_mc):
                    output = self.unet(batch)
                    output, logvar = torch.chunk(output, 2, dim=1)
                    output = output.squeeze(1)
                    logits.append(output)
                    logvar = logvar.squeeze()
                    uncert = torch.exp(logvar)
                    stds.append(uncert)

                logits = torch.stack(logits)
                mean_logits = logits.mean(dim=0)
                epistemic_uncertainty = logits.var(dim=0)
                output = torch.round(torch.sigmoid(mean_logits)).byte()

                stds = torch.stack(stds)
                aleatoric_uncertainty = stds.mean(dim=0)
                uncert = epistemic_uncertainty + aleatoric_uncertainty
                if y_pad != 0 and x_pad != 0:
                    if j == len(dl) - 1:
                        preds[j * self.batch_size :] = output[
                            :, y_pad:-y_pad, x_pad:-x_pad
                        ]
                        uncerts[j * self.batch_size :] = uncert[
                            :, y_pad:-y_pad, x_pad:-x_pad
                        ]
                    else:
                        preds[j * batch.shape[0] : (j + 1) * batch.shape[0]] = output[
                            :, y_pad:-y_pad, x_pad:-x_pad
                        ]
                        uncerts[j * batch.shape[0] : (j + 1) * batch.shape[0]] = uncert[
                            :, y_pad:-y_pad, x_pad:-x_pad
                        ]
                elif y_pad != 0:
                    if j == len(dl) - 1:
                        preds[j * self.batch_size :] = output[:, y_pad:-y_pad]
                        uncerts[j * self.batch_size :] = uncert[:, y_pad:-y_pad]
                    else:
                        preds[j * batch.shape[0] : (j + 1) * batch.shape[0]] = output[
                            :, y_pad:-y_pad
                        ]
                        uncerts[j * batch.shape[0] : (j + 1) * batch.shape[0]] = uncert[
                            :, y_pad:-y_pad
                        ]
                elif x_pad != 0:
                    if j == len(dl) - 1:
                        preds[j * self.batch_size :] = output[:, :, x_pad:-x_pad]
                        uncerts[j * self.batch_size :] = uncert[:, :, x_pad:-x_pad]
                    else:
                        preds[j * batch.shape[0] : (j + 1) * batch.shape[0]] = output[
                            :, :, x_pad:-x_pad
                        ]
                        uncerts[j * batch.shape[0] : (j + 1) * batch.shape[0]] = uncert[
                            :, :, x_pad:-x_pad
                        ]
                else:
                    if j == len(dl) - 1:
                        preds[j * self.batch_size :] = output
                        uncerts[j * self.batch_size :] = uncert
                    else:
                        preds[j * batch.shape[0] : (j + 1) * batch.shape[0]] = output
                        uncerts[j * batch.shape[0] : (j + 1) * batch.shape[0]] = uncert

        return preds, uncerts

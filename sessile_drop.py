import pathlib
import pickle
from dataclasses import asdict
from typing import Dict, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import requests
import scipy.ndimage as ndi
import scipy.signal as sig
import torch
from scipy.optimize import curve_fit
from torchvision.transforms import v2
from tqdm import tqdm

from contact_angle import contact_angle
from schema import Results
from seq2seq import DecoderRNN, EncoderRNN, Seq2SeqInference
from unet import UNet, UNetInference
from utils import (
    init_frame,
    liquid_absorption,
    liquid_absorption_uncertainty,
    power_law,
    px_size,
    schott,
    swelling,
    swelling_uncertainty,
    vid_to_tensor,
)


def absorption_correction(
    surface_area: torch.tensor, t_ai_idx: int, t_af_idx: int, t: np.ndarray
) -> int:
    absorption = surface_area[t_ai_idx : t_af_idx + 1]
    t_abs = t[t_ai_idx : t_af_idx + 1] - t[t_ai_idx]

    second_deriv = np.gradient(np.gradient(absorption, t_abs), t_abs)

    second_deriv_max = np.where(second_deriv == np.max(second_deriv))[0][0]
    min_absorption_idx = np.where(absorption == np.min(absorption))[0][0]

    if (second_deriv_max < 0.7 * absorption.shape[0]) and (absorption.shape[0] > 50):
        return t_af_idx
    else:
        while second_deriv_max > min_absorption_idx:
            # new_second_deriv = np.gradient(np.gradient(absorption[:second_deriv_max], t_abs[:second_deriv_max]), t_abs[:second_deriv_max])

            # second_deriv_max = np.where(new_second_deriv == np.max(new_second_deriv))[0][0]
            second_deriv_max -= 1
        return second_deriv_max + t_ai_idx - 1


def swelling_correction(
    surface_area: torch.tensor, t_af_idx: int, t_si_idx: int
) -> int:
    unc_sw = surface_area[t_af_idx:]

    min_sa_idx = np.where(unc_sw == np.min(unc_sw))[0][0]
    si_idx = np.where(unc_sw == unc_sw[t_si_idx - t_af_idx])[0][0]

    while (unc_sw[si_idx] - unc_sw[min_sa_idx]) > 0.1 * unc_sw[min_sa_idx]:
        si_idx -= 1

    return si_idx + t_af_idx + 1


def surface_area(
    frame: torch.tensor,
    mask: torch.tensor,
    uncerts: torch.tensor,
    pixel_size: float,
    device: Union[torch.device, str, int],
) -> float:
    """
    This function is used to find the surface area of the droplet in mm^2 given
    that the pixel size is given in mm.

    Parameters
    ----------
    frame : torch.tensor
        The segmentation mask to find the surface area of the droplet in.
    mask : torch.tensor
        The mask to remove the tablet and leave only the droplet.
    uncerts : torch.tensor
        The uncertainties in the masks.
    pixel_size : float
        The size of a pixel in mm.

    Returns
    -------
     : float
        The surface area of the droplet in mm^2.
    """
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device(f"cuda:{device}")
    else:
        device = device

    drop = frame - mask - mask
    drop[drop < 0] = 0
    labels = torch.tensor(ndi.label(drop.cpu().numpy())[0], device=device)
    y_drop, x_drop = torch.where(
        labels == torch.topk(torch.bincount(labels.flatten()), 2)[1][-1]
    )
    drop_img = torch.zeros_like(labels, device=device)
    drop_img[y_drop, x_drop] = 1
    drop_img = torch.tensor(
        ndi.binary_fill_holes(drop_img.cpu().numpy()), device=device
    )
    sas_uncert = torch.sqrt(uncerts[y_drop, x_drop].sum()) * pixel_size**2

    return drop_img, drop_img.sum() * pixel_size**2, sas_uncert


def regime_detection(
    surface_area: torch.tensor,
    surface_area_uncert: torch.tensor,
    ca_left: torch.tensor,
    ca_right: torch.tensor,
    ca_avg: torch.tensor,
    fps: int,
    enc: EncoderRNN,
    dec: DecoderRNN,
    device: Union[torch.device, str, int],
    seq2seq_path: Union[pathlib.Path, str],
) -> Results:
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device(f"cuda:{device}")
    else:
        device = device

    seq2seq_path = (
        pathlib.Path(seq2seq_path) if isinstance(seq2seq_path, str) else seq2seq_path
    )

    filt_sas = ndi.gaussian_filter1d(surface_area.cpu().numpy(), 3)
    surface_area_uncert = surface_area_uncert.cpu().numpy()
    t = np.linspace(0, filt_sas.shape[0] / fps, filt_sas.shape[0])
    try:
        # t_ai_idx = sig.find_peaks(filt_sas, height=(0.3*filt_sas.max().item(),
        # 0.85*filt_sas.max().item()))[0][0]
        t_ai_idx = sig.find_peaks(filt_sas, height=0.3 * filt_sas.max().item())[0][0]
    except IndexError:
        raise IndexError("Straight to swelling!")

    seq2seq = Seq2SeqInference(enc, dec, 1, device_id=device)
    seq2seq.load_checkpoint(seq2seq_path)
    seq2seq.load_data([filt_sas[t_ai_idx:] / filt_sas[t_ai_idx:].max()])
    labels, _ = seq2seq.chic_actress()
    labels = labels.pop()
    changepoints = torch.where(
        labels.diff() != 0
    )[
        0
    ]  # changed from == 1 to != 0 because == 1 will never detect changepoints when there is only absorption and swelling whereas the difference being != 0 will
    if labels.unique().shape[0] != labels.unique_consecutive().shape[0]:
        if labels.unique().shape[0] == 3:
            changepoints = torch.tensor(
                [changepoints[0], changepoints[-1]], device=device
            )
        elif labels.unique().shape[0] == 2:
            changepoints = torch.tensor([changepoints[0]], device=device)
    if changepoints.shape[0] == 2:
        t_af_idx, t_si_idx = (
            changepoints  # the two points where the labels change are times when absorption ends and swelling begins, this assumes that there is always liquid absorption, transition and swelling
        )
        t_af_idx = t_af_idx.cpu().numpy() + t_ai_idx
        t_si_idx = t_si_idx.cpu().numpy() + t_ai_idx
        t_af_idx = absorption_correction(
            filt_sas, t_ai_idx, t_af_idx, t
        )  # correction to absorption profiles that are overestimate by the S2S model
        t_si_idx = swelling_correction(filt_sas, t_af_idx, t_si_idx)
        t_ai = t_ai_idx / fps  # the liquid absorption start time
        t_af = t[t_af_idx]  # final absorption time corresponds to the first peak's time
        t_si = t[
            t_si_idx
        ]  # initial swelling time corresponds to the second peak's time

        ca_left = ca_left[t_ai_idx : t_af_idx + 1].cpu().numpy()
        ca_right = ca_right[t_ai_idx : t_af_idx + 1].cpu().numpy()
        ca_avg = ca_avg[t_ai_idx : t_af_idx + 1].cpu().numpy()

        indices = lambda z: z.nonzero()[0]
        ca_left_nans = np.isnan(ca_left)
        ca_right_nans = np.isnan(ca_right)
        ca_avg_nans = np.isnan(ca_avg)

        if indices(ca_left_nans).shape[0] != 0:
            if len(indices(~ca_left_nans)) > len(indices(ca_left_nans)):
                ca_left[ca_left_nans] = np.interp(
                    indices(ca_left_nans),
                    indices(~ca_left_nans),
                    ca_left[~ca_left_nans],
                )
            else:
                # if there are more NaNs than not in the contact angle data then
                # the ellipse fitting cannot be trusted and we can't get contact
                # angle information from this so return 0s
                ca_left = np.zeros(t_af_idx + 1 - t_ai_idx)
        if indices(ca_right_nans).shape[0] != 0:
            if len(indices(~ca_right_nans)) > len(indices(ca_right_nans)):
                ca_right[ca_right_nans] = np.interp(
                    indices(ca_right_nans),
                    indices(~ca_right_nans),
                    ca_right[~ca_right_nans],
                )
            else:
                ca_right = np.zeros(t_af_idx + 1 - t_ai_idx)
        if indices(ca_avg_nans).shape[0] != 0:
            if len(indices(~ca_avg_nans)) > len(indices(ca_avg_nans)):
                ca_avg[ca_avg_nans] = np.interp(
                    indices(ca_avg_nans), indices(~ca_avg_nans), ca_avg[~ca_avg_nans]
                )
            else:
                ca_avg = np.zeros(t_af_idx + 1 - t_ai_idx)

        filt_casL = ndi.gaussian_filter1d(ca_left, 3)
        filt_casR = ndi.gaussian_filter1d(ca_right, 3)
        filt_casAvg = ndi.gaussian_filter1d(ca_avg, 3)

        la = liquid_absorption(filt_sas[t_ai_idx : t_af_idx + 1])
        la_uncerts = liquid_absorption_uncertainty(
            filt_sas[t_ai_idx : t_af_idx + 1],
            surface_area_uncert[t_ai_idx : t_af_idx + 1],
        )
        sw = swelling(filt_sas[t_si_idx:])
        sw_uncerts = swelling_uncertainty(surface_area_uncert[t_si_idx:])
        t_la = t[t_ai_idx : t_af_idx + 1] - t_ai
        t_sw = (
            t[t_si_idx:] - t_si
        )  # the swelling times for fitting are shifted to start from 0

        popt_la, pcov_la = curve_fit(
            power_law, t_la, la, sigma=la_uncerts, absolute_sigma=True
        )
        k, m = popt_la
        k_err, m_err = np.sqrt(np.diagonal(pcov_la))
        try:
            popt_sw, pcov_sw = curve_fit(
                schott,
                t_sw,
                sw,
                sigma=sw_uncerts,
                absolute_sigma=True,
                p0=(1.1, sw[-1], 0.0),
                bounds=([0.1, sw[-1], 0.0], [np.inf, 100, 10.0]),
            )  # Schott model has an asymptote when all parameters = 1 so need to start one of them from not 1
        except RuntimeError:
            popt_sw, pcov_sw = curve_fit(
                schott,
                t_sw,
                sw,
                sigma=sw_uncerts,
                absolute_sigma=True,
                p0=(1.1, sw[-1], 0.0),
                bounds=([0.1, sw[-1], 0.0], [np.inf, 100, 10.0]),
            )
        A, siginf, tr = popt_sw
        A_err, siginf_err, tr_err = np.sqrt(np.diagonal(pcov_sw))

        t_ai = t_ai
        t_af = t_af
        t_si = t_si
        t_ai_idx = t_ai_idx
        t_af_idx = t_af_idx
        t_si_idx = t_si_idx
    elif changepoints.shape[0] == 1 and torch.equal(
        torch.unique(labels), torch.tensor([0, 2], device=device)
    ):
        t_af_idx = changepoints[0]
        t_si_idx = t_af_idx + 1
        t_af_idx = t_af_idx.cpu().numpy() + t_ai_idx
        t_si_idx = t_si_idx.cpu().numpy() + t_ai_idx
        t_af_idx = absorption_correction(
            filt_sas, t_ai_idx, t_af_idx, t
        )  # correction to absorption profiles that are overestimate by the S2S model
        t_si_idx = swelling_correction(filt_sas, t_af_idx, t_si_idx)
        t_ai = t_ai_idx / fps  # the liquid absorption start time
        t_af = t[t_af_idx]  # final absorption time corresponds to the first peak's time
        t_si = t[
            t_si_idx
        ]  # initial swelling time corresponds to the second peak's time

        ca_left = ca_left[t_ai_idx : t_af_idx + 1].cpu().numpy()
        ca_right = ca_right[t_ai_idx : t_af_idx + 1].cpu().numpy()
        ca_avg = ca_avg[t_ai_idx : t_af_idx + 1].cpu().numpy()

        indices = lambda z: z.nonzero()[0]
        ca_left_nans = np.isnan(ca_left)
        ca_right_nans = np.isnan(ca_right)
        ca_avg_nans = np.isnan(ca_avg)

        if indices(ca_left_nans).shape[0] != 0:
            if len(indices(~ca_left_nans)) > len(indices(ca_left_nans)):
                ca_left[ca_left_nans] = np.interp(
                    indices(ca_left_nans),
                    indices(~ca_left_nans),
                    ca_left[~ca_left_nans],
                )
            else:
                # if there are more NaNs than not in the contact angle data then
                # the ellipse fitting cannot be trusted and we can't get contact
                # angle information from this so return 0s
                ca_left = np.zeros(t_af_idx + 1 - t_ai_idx)
        if indices(ca_right_nans).shape[0] != 0:
            if len(indices(~ca_right_nans)) > len(indices(ca_right_nans)):
                ca_right[ca_right_nans] = np.interp(
                    indices(ca_right_nans),
                    indices(~ca_right_nans),
                    ca_right[~ca_right_nans],
                )
            else:
                ca_right = np.zeros(t_af_idx + 1 - t_ai_idx)
        if indices(ca_avg_nans).shape[0] != 0:
            if len(indices(~ca_avg_nans)) > len(indices(ca_avg_nans)):
                ca_avg[ca_avg_nans] = np.interp(
                    indices(ca_avg_nans), indices(~ca_avg_nans), ca_avg[~ca_avg_nans]
                )
            else:
                ca_avg = np.zeros(t_af_idx + 1 - t_ai_idx)

        filt_casL = ndi.gaussian_filter1d(ca_left, 3)
        filt_casR = ndi.gaussian_filter1d(ca_right, 3)
        filt_casAvg = ndi.gaussian_filter1d(ca_avg, 3)

        la = liquid_absorption(filt_sas[t_ai_idx : t_af_idx + 1])
        la_uncerts = liquid_absorption_uncertainty(
            filt_sas[t_ai_idx : t_af_idx + 1],
            surface_area_uncert[t_ai_idx : t_af_idx + 1],
        )
        sw = swelling(filt_sas[t_si_idx:])
        sw_uncerts = swelling_uncertainty(surface_area_uncert[t_si_idx:])
        t_la = t[t_ai_idx : t_af_idx + 1] - t_ai
        t_sw = (
            t[t_si_idx:] - t_si
        )  # the swelling times for fitting are shifted to start from 0

        popt_la, pcov_la = curve_fit(
            power_law, t_la, la, sigma=la_uncerts, absolute_sigma=True
        )
        k, m = popt_la
        k_err, m_err = np.sqrt(np.diagonal(pcov_la))
        try:
            popt_sw, pcov_sw = curve_fit(
                schott,
                t_sw,
                sw,
                sigma=sw_uncerts,
                absolute_sigma=True,
                p0=(1.1, sw[-1], 0.0),
                bounds=([0.1, sw[-1], 0.0], [np.inf, 100, 10.0]),
            )  # Schott model has an asymptote when all parameters = 1 so need to start one of them from not 1
        except RuntimeError:
            popt_sw, pcov_sw = curve_fit(
                schott,
                t_sw,
                sw,
                sigma=sw_uncerts,
                absolute_sigma=True,
                p0=(1.1, sw[-1], 0.0),
                bounds=([0.1, sw[-1], 0.0], [np.inf, 100, 10.0]),
            )
        A, siginf, tr = popt_sw
        A_err, siginf_err, tr_err = np.sqrt(np.diagonal(pcov_sw))

        t_ai = t_ai
        t_af = t_af
        t_si = t_si
        t_ai_idx = t_ai_idx
        t_af_idx = t_af_idx
        t_si_idx = t_si_idx
    elif changepoints.shape[0] == 1 and torch.equal(
        torch.unique(labels), torch.tensor([0, 1], device=device)
    ):
        t_af_idx = changepoints[0]
        t_af_idx = t_af_idx.cpu().numpy() + t_ai_idx
        t_af_idx = absorption_correction(
            filt_sas, t_ai_idx, t_af_idx, t
        )  # correction to absorption profiles that are overestimate by the S2S model
        t_ai = t_ai_idx / fps
        t_af = t[t_af_idx]

        ca_left = ca_left[t_ai_idx : t_af_idx + 1].cpu().numpy()
        ca_right = ca_right[t_ai_idx : t_af_idx + 1].cpu().numpy()
        ca_avg = ca_avg[t_ai_idx : t_af_idx + 1].cpu().numpy()

        indices = lambda z: z.nonzero()[0]
        ca_left_nans = np.isnan(ca_left)
        ca_right_nans = np.isnan(ca_right)
        ca_avg_nans = np.isnan(ca_avg)

        if indices(ca_left_nans).shape[0] != 0:
            if len(indices(~ca_left_nans)) > len(indices(ca_left_nans)):
                ca_left[ca_left_nans] = np.interp(
                    indices(ca_left_nans),
                    indices(~ca_left_nans),
                    ca_left[~ca_left_nans],
                )
            else:
                # if there are more NaNs than not in the contact angle data then
                # the ellipse fitting cannot be trusted and we can't get contact
                # angle information from this so return 0s
                ca_left = np.zeros(t_af_idx + 1 - t_ai_idx)
        if indices(ca_right_nans).shape[0] != 0:
            if len(indices(~ca_right_nans)) > len(indices(ca_right_nans)):
                ca_right[ca_right_nans] = np.interp(
                    indices(ca_right_nans),
                    indices(~ca_right_nans),
                    ca_right[~ca_right_nans],
                )
            else:
                ca_right = np.zeros(t_af_idx + 1 - t_ai_idx)
        if indices(ca_avg_nans).shape[0] != 0:
            if len(indices(~ca_avg_nans)) > len(indices(ca_avg_nans)):
                ca_avg[ca_avg_nans] = np.interp(
                    indices(ca_avg_nans), indices(~ca_avg_nans), ca_avg[~ca_avg_nans]
                )
            else:
                ca_avg = np.zeros(t_af_idx + 1 - t_ai_idx)

        filt_casL = ndi.gaussian_filter1d(ca_left, 3)
        filt_casR = ndi.gaussian_filter1d(ca_right, 3)
        filt_casAvg = ndi.gaussian_filter1d(ca_avg, 3)

        la = liquid_absorption(filt_sas[t_ai_idx : t_af_idx + 1])
        la_uncerts = liquid_absorption_uncertainty(
            filt_sas[t_ai_idx : t_af_idx + 1],
            surface_area_uncert[t_ai_idx : t_af_idx + 1],
        )
        t_la = t[t_ai_idx : t_af_idx + 1] - t_ai

        popt_la, pcov_la = curve_fit(
            power_law, t_la, la, sigma=la_uncerts, absolute_sigma=True
        )
        k, m = popt_la
        k_err, m_err = np.sqrt(np.diagonal(pcov_la))

        t_si_idx = 0
        t_si = 0
        A, siginf, tr = 0.0, 0.0, 0.0
        A_err, siginf_err, tr_err = 0.0, 0.0, 0.0

        t_ai = t_ai
        t_af = t_af
        t_ai_idx = t_ai_idx
        t_af_idx = t_af_idx
    else:
        t_af_idx = -1
        t_ai = t_ai_idx / fps
        t_af = t[t_af_idx]

        ca_left = ca_left[t_ai_idx:].cpu().numpy()
        ca_right = ca_right[t_ai_idx:].cpu().numpy()
        ca_avg = ca_avg[t_ai_idx:].cpu().numpy()

        indices = lambda z: z.nonzero()[0]
        ca_left_nans = np.isnan(ca_left)
        ca_right_nans = np.isnan(ca_right)
        ca_avg_nans = np.isnan(ca_avg)

        if indices(ca_left_nans).shape[0] != 0:
            if len(indices(~ca_left_nans)) > len(indices(ca_left_nans)):
                ca_left[ca_left_nans] = np.interp(
                    indices(ca_left_nans),
                    indices(~ca_left_nans),
                    ca_left[~ca_left_nans],
                )
            else:
                # if there are more NaNs than not in the contact angle data then
                # the ellipse fitting cannot be trusted and we can't get contact
                # angle information from this so return 0s
                ca_left = np.zeros(t_af_idx + 1 - t_ai_idx)
        if indices(ca_right_nans).shape[0] != 0:
            if len(indices(~ca_right_nans)) > len(indices(ca_right_nans)):
                ca_right[ca_right_nans] = np.interp(
                    indices(ca_right_nans),
                    indices(~ca_right_nans),
                    ca_right[~ca_right_nans],
                )
            else:
                ca_right = np.zeros(t_af_idx + 1 - t_ai_idx)
        if indices(ca_avg_nans).shape[0] != 0:
            if len(indices(~ca_avg_nans)) > len(indices(ca_avg_nans)):
                ca_avg[ca_avg_nans] = np.interp(
                    indices(ca_avg_nans), indices(~ca_avg_nans), ca_avg[~ca_avg_nans]
                )
            else:
                ca_avg = np.zeros(t_af_idx + 1 - t_ai_idx)

        filt_casL = ndi.gaussian_filter1d(ca_left, 3)
        filt_casR = ndi.gaussian_filter1d(ca_right, 3)
        filt_casAvg = ndi.gaussian_filter1d(ca_avg, 3)

        la = liquid_absorption(filt_sas[t_ai_idx:])
        la_uncerts = liquid_absorption_uncertainty(
            filt_sas[t_ai_idx:], surface_area_uncert[t_ai_idx:]
        )
        t_la = t[t_ai_idx:] - t_ai

        popt_la, pcov_la = curve_fit(
            power_law, t_la, la, sigma=la_uncerts, absolute_sigma=True
        )
        k, m = popt_la
        k_err, m_err = pcov_la

        t_si_idx = 0
        t_si = 0
        A, siginf, tr = 0.0, 0.0, 0.0
        A_err, siginf_err, tr_err = 0.0, 0.0, 0.0

        t_ai = t_ai
        t_ai_idx = t_ai_idx
        t_af = t_af
        t_af_idx = t_af_idx

    results = Results(
        sas=filt_sas,
        sas_uncert=surface_area_uncert,
        casL=filt_casL,
        casR=filt_casR,
        casAvg=filt_casAvg,
        t_ai=t_ai,
        t_af=t_af,
        t_si=t_si,
        k=k,
        m=m,
        A=A,
        siginf=siginf,
        tr=tr,
        t_ai_idx=t_ai_idx,
        t_af_idx=t_af_idx,
        t_si_idx=t_si_idx,
        k_err=k_err,
        m_err=m_err,
        A_err=A_err,
        siginf_err=siginf_err,
        tr_err=tr_err,
    )

    return results


def sessile_drop(
    path: Union[str, pathlib.Path],
    device: Union[torch.device, str, int],
    needle_size: float,
    row_idx: int,
    batch_size: int,
    unet: UNet,
    unet_path: Union[pathlib.Path, str],
    fps: int,
    enc: EncoderRNN,
    dec: DecoderRNN,
    seq2seq_path: Union[pathlib.Path, str],
) -> Tuple[torch.tensor, Results]:
    """
    This function is used to analyse an entire sessile drop video, extract the
    surface area and contact angles over time, find the liquid absorption,
    transition and swelling regimes and also fit the power law and Schott models
    to the liquid absorption and swelling data, respectively. This returns the
    surface area of the droplet, the contact angle on the left, right and
    average during liquid absorption, the times of the end of liquid absorption
    and start of swelling and the fitting parameters for the power law and
    Schott models.

    Parameters
    ----------
    path : pathlib.Path or str
        The path to the Zarr file containing the video.
    device : torch.device or str
        The device to do the calculations on.
    needle_size : float
        The physical size of the needle used as a dropper in mm.
    batch_size : int
        The batch size to analyse the frames concurrently.
    unet : torch.nn.Module
        The U-Net model for segmenting frames.
    unet_path : pathlib.Path or str
        The path to the trained U-Net model.
    fps : int
        The frames per second (FPS) of the video to calculate times with.
    enc : torch.nn.Module
        The encoder for the Seq2Seq model.
    dec : torch.nn.Module
        The decoder for the Seq1Seq model.
    seq2seq_path : pathlib.Path or str
        The path to the trained Seq2Seq model for regime separation.

    Returns
    -------
    preds : torch.tensor
        The U-Net predictions.
    results: schema.Results
        The dataclass containing the results of the analysis: this object
        contains the surface area over the entire video (including pre-drop);
        the left, right and average contact angles during the
        absorption-dominated phase if it exists; the initial and final
        absorption and initial swelling times and their respective indices; and
        the fitting parameters from the power law and Schott models.
    """
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device(f"cuda:{device}")
    else:
        device = device

    path = pathlib.Path(path) if isinstance(path, str) else path
    unet_path = pathlib.Path(unet_path) if isinstance(unet_path, str) else unet_path
    seq2seq_path = (
        pathlib.Path(seq2seq_path) if isinstance(seq2seq_path, str) else seq2seq_path
    )
    data = vid_to_tensor(path, device="cpu")

    pixel_size = px_size(data[0], needle_size, row_idx=row_idx)

    data = v2.functional.to_dtype(data, torch.float, scale=True)

    casR = torch.zeros(data.shape[0])
    casL = torch.zeros(data.shape[0])
    casAvg = torch.zeros(data.shape[0])
    sas = torch.zeros(data.shape[0])
    sas_uncerts = torch.zeros(data.shape[0])

    unet_inference = UNetInference(unet, batch_size, device, n_mc=3)
    unet_inference.load_checkpoint(unet_path)
    preds, uncerts = unet_inference.castle_valet(data)
    tab_img, tab_grad, tab_intercept, p1, p2 = init_frame(
        preds[0], data[0], device=device
    )

    for jj in tqdm(range(preds.shape[0])):
        try:
            preds[jj], sas[jj], sas_uncerts[jj] = surface_area(
                preds[jj], tab_img, uncerts[jj], pixel_size, device
            )
            casL[jj], casR[jj], casAvg[jj] = contact_angle(
                preds[jj], tab_grad, tab_intercept, device=device
            )
        except:
            continue

    results = regime_detection(
        sas, sas_uncerts, casL, casR, casAvg, fps, enc, dec, device, seq2seq_path
    )

    return preds, uncerts, results


def sessile_drop_save(
    masks: npt.NDArray,
    uncerts: npt.NDArray,
    results: Results,
    video_path: Union[str, pathlib.Path],
    fps: int,
    save_masks: bool = False,
    save_dir: Optional[Union[str, pathlib.Path]] = None,
    send_to_db: bool = False,
    db_kwargs: Optional[Dict] = None,
) -> None:
    """
    Function for saving results of video analysis.

    Parameters
    ----------
    masks: numpy.ndarray
        The masks for each frame of the video.
    results: schema.Results
        The dataclass containing the results of the analysis: this object
        contains the surface area over the entire video (including pre-drop);
        the left, right and average contact angles during the
        absorption-dominated phase if it exists; the initial and final
        absorption and initial swelling times and their respective indices; and
        the fitting parameters from the power law and Schott models.
    video_path: str or pathlib.Path
        The path to the video.
    fps : int
        The frames per second (FPS) of the video.
    save_masks: bool, optional
        Whether or not to save the segmentation masks. Default is False.
    save_dir: str or pathlib.Path, optional
        Where to save the results. Default is None.
    """
    video_path = pathlib.Path(video_path) if isinstance(video_path, str) else video_path
    if save_dir:
        save_dir = pathlib.Path(save_dir) if isinstance(save_dir, str) else save_dir
    else:
        save_dir = video_path.parent

    if save_masks:
        with open(save_dir / (video_path.name[:-4] + "_masks.pkl"), "wb") as f:
            pickle.dump(masks, f)
        with open(save_dir / (video_path.name[:-4] + "_masks_uncerts.pkl"), "wb") as f:
            pickle.dump(uncerts, f)
    with open(save_dir / (video_path.name[:-4] + ".pkl"), "wb") as f:
        pickle.dump(results, f)

    params_list = [
        "t_ai",
        "t_af",
        "t_si",
        "k",
        "m",
        "A",
        "siginf",
        "t_r",
        "k_err",
        "m_err",
        "A_err",
        "siginf_err",
        "tr_err",
    ]
    params_values = [
        results.t_ai,
        results.t_af,
        results.t_si,
        results.k,
        results.m,
        results.A,
        results.siginf,
        results.tr,
        results.k_err,
        results.m_err,
        results.A_err,
        results.siginf_err,
        results.tr_err,
    ]
    params_units = [
        "s",
        "s",
        "s",
        "",
        "",
        "s/mm^2",
        "mm^2",
        "s",
        "",
        "",
        "s/mm^2",
        "mm^2",
        "s",
    ]

    params_df = pd.DataFrame(
        {"Parameter": params_list, "Values": params_values, "Units": params_units}
    )
    t = np.linspace(0, results.sas.shape[0] / fps, int(results.sas.shape[0]))
    t_ai_idx = results.t_ai_idx
    t_af_idx = results.t_af_idx
    t_si_idx = results.t_si_idx
    t_sa = t[t_ai_idx:] - t[t_ai_idx]
    t_abs = t[t_ai_idx : t_af_idx + 1] - t[t_ai_idx]
    if t_af_idx == -1:
        t_abs = t[t_ai_idx:] - t[t_ai_idx]
    t_sw = t[t_si_idx:] - t[t_si_idx]

    sa_df = pd.DataFrame(
        {"t": t_sa, "S": results.sas[t_ai_idx:], "dS": results.sas_uncert[t_ai_idx:]}
    )
    ca_df = pd.DataFrame(
        {
            "t": t_abs,
            "phi_L": results.casL,
            "phi_R": results.casR,
            "phi_Avg": results.casAvg,
        }
    )
    if t_af_idx == -1:
        la_df = pd.DataFrame(
            {
                "t": t_abs,
                "y": liquid_absorption(results.sas[t_ai_idx:]),
                "dy": liquid_absorption_uncertainty(
                    results.sas[t_ai_idx:], results.sas_uncert[t_ai_idx:]
                ),
            }
        )
    else:
        la_df = pd.DataFrame(
            {
                "t": t_abs,
                "y": liquid_absorption(results.sas[t_ai_idx : t_af_idx + 1]),
                "dy": liquid_absorption_uncertainty(
                    results.sas[t_ai_idx : t_af_idx + 1],
                    results.sas_uncert[t_ai_idx : t_af_idx + 1],
                ),
            }
        )
    sw_df = pd.DataFrame(
        {
            "t": t_sw,
            "sigma": swelling(results.sas[t_si_idx:]),
            "dsigma": swelling_uncertainty(results.sas_uncert[t_si_idx:]),
        }
    )

    writer = pd.ExcelWriter(
        save_dir / (video_path.name[:-4] + ".xlsx"), engine="openpyxl"
    )
    params_df.to_excel(writer, sheet_name="Parameters", index=False)
    sa_df.to_excel(writer, sheet_name="Surface Area", index=False)
    ca_df.to_excel(writer, sheet_name="Contact Angle", index=False)
    la_df.to_excel(writer, sheet_name="Absorption", index=False)
    sw_df.to_excel(writer, sheet_name="Swelling", index=False)
    writer.close()

    if send_to_db:
        results_dict = asdict(results)

        for k, v in results_dict.items():
            if isinstance(v, np.ndarray):
                results_dict[k] = v.tolist()
            elif isinstance(v, np.integer):
                results_dict[k] = int(v)

        if db_kwargs is not None:
            db_kwargs.update({"results": results_dict})

        requests.post("http://10.5.49.6:8000/create-sessile-drop", json=db_kwargs)

    print("Data saved successfully!")

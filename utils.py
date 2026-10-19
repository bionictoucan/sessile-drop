import numpy as np
import scipy.ndimage as ndi
from skimage.feature import canny
from skimage.filters import threshold_minimum
from skimage.segmentation import join_segmentations
import zarr, torch, pathlib, numcodecs, pims, platform
from typing import Union, Tuple
from tqdm import tqdm
import pickle
import numpy.typing as npt

if not platform.processor() == "aarch64":
    from imagecodecs.numcodecs import Jpegxl
    numcodecs.register_codec(Jpegxl)


def vid_to_tensor(
    path: Union[pathlib.Path, str], device: Union[torch.device, str]
) -> torch.tensor:
    """
    This function loads the entire video file from memory to the device that
    inference will be carried out on.

    Parameters
    ----------
    path : pathlib.Path or str
        The path to the video file.
    device : torch.device or str
        The device to load the data on to.

    Returns
    -------
    tensor_data : torch.tensor
        The torch tensor containing all of the video data.
    """
    if path.suffix == ".avi" or path.suffix == ".mp4":
        data = pims.Video(path)
    elif path.suffix == ".pkl":
        data = pickle.load(open(path, "rb"))
        data = pims.Video(data)
    else:
        data = zarr.open(path, mode="r")
    tensor_data = torch.zeros(data.shape, device=device, dtype=torch.uint8)
    for j in tqdm(range(data.shape[0]), desc="Loading Data: "):
        tensor_data[j] = torch.tensor(data[j], dtype=torch.uint8, device=device)

    return tensor_data


def px_size(frame: torch.tensor, needle_size: float, row_idx: int) -> float:
    """
    This function works out the physical size of a pixel in mm based on the
    outer diameter of the needle and a frame from the video.

    Parameters
    ----------
    frame : torch.tensor
        The frame from the video.
    needle_size : float
        The size of the outer diameter of the needle in mm.

    Returns
    -------
     : float
        The pixel size in mm.
    """

    edges = torch.tensor(canny(frame[:,:,0].cpu().numpy(), sigma=3))

    ind = torch.where(
        edges[row_idx]
    )[0]  # row_idx = 20 for main lab, row_idx = 650 for DataFactory
    if (ind[-1] - ind[0]) < 100:
        needle_px = (ind[-1]-ind[0]).item()
    else:
        needle_px = (ind[1]-ind[0]).item()

    return needle_size / needle_px

def init_frame(
    frame: torch.tensor, data: torch.tensor, device: Union[torch.device, str, int]
) -> Tuple[torch.tensor, float, torch.tensor, torch.tensor]:
    """
    Analyse the initial frame to return the mask containing the tablet, the
    gradient of the surface of the tablet and the two points used to calculate
    the gradient of the surface of the tablet.

    Parameters
    ----------
    frame : torch.tensor
        The initial frame to analyse.

    Returns
    -------
    tab_img : torch.tensor
        The mask of the tablet.
    m : float
        The gradient of the surface of the tablet.
    c : float
        The intercept of the surface of the tablet.
    point1 : torch.tensor
        The point on the left-side of the tablet used for calculating the
        gradient of the surface.
    point2 : torch.tensor
        The point on the right-side of the tablet used for calculating the
        gradient of the surface.
    """
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device(f"cuda:{device}")

    thresh = threshold_minimum(data.cpu().numpy())
    binary = data.cpu().numpy() < thresh
    joint_seg = join_segmentations(binary[:,:,0].astype(int), frame.cpu().numpy())
    labels = torch.tensor(ndi.label(joint_seg)[0], device=device)
    tab_img = torch.zeros_like(labels, device=device)
    y_tablet, x_tablet = torch.where(
        labels == torch.topk(torch.bincount(labels.flatten()), 2)[1][-1]
    )
    tab_img[y_tablet, x_tablet] = 1
    yy, xx = torch.where(tab_img == 1)
    x1 = torch.topk(torch.unique(xx), 10)[0][-1]
    y1 = yy[xx == x1].min()
    x2 = torch.topk(torch.unique(xx), 10, largest=False)[0][-1]
    y2 = yy[xx == x2].min()
    point1 = torch.tensor([y1, x1])
    point2 = torch.tensor([y2, x2])
    m = torch.divide(*(point2 - point1))
    c = point1[0] - m*point1[1]

    return tab_img, m, c, point1, point2


def liquid_absorption(S: npt.NDArray) -> npt.NDArray:
    """
    A function to calculate the liquid absorption fraction of the surface area
    of the droplet during the liquid absorption phase, following the equation

    .. math::
        y (t) = \left | \frac{S(t) - S(t_{a,i})}{S(t_{a,i})}

    where :math:`t_{a,i}` is the initial absorption time.

    Parameters
    ----------
    S : numpy.ndarray
        The surface area profile for liquid absorption.

    Returns
    -------
     : numpy.ndarray
        The liquid absorption.
    """

    return np.abs(S - S[0]) / S[0]

def liquid_absorption_uncertainty(S: npt.NDArray, S_uncert: npt.NDArray) -> npt.NDArray:
    dydS = 1. / S[0]
    dydS0 = - S / S[0]**2

    var_y = dydS**2 * S_uncert**2 + (dydS0**2 * S_uncert[0]**2)

    return np.sqrt(var_y)

def swelling(S: npt.NDArray) -> npt.NDArray:
    """
    A function to calculate the swelling of the surface area of the droplet
    during the swelling phase, following the equation

    .. math::
        \sigma(t) = S(t) - S(t_{s,i})

    where :math:`t_{s,i}` is the initial swelling time.

    Parameters
    ----------
    S : numpy.ndarray
        The surface area profile for swelling.

    Returns
    -------
     : numpy.ndarray
        The swelling.
    """
    return S - S[0]

def swelling_uncertainty(S_uncert: npt.NDArray) -> npt.NDArray:
    return np.sqrt(S_uncert**2 + S_uncert[0]**2)


def power_law(t: npt.NDArray, k: float, m: float) -> npt.NDArray:
    """
    The power-law model to be fitted to the liquid absorption profile following
    the equation

    .. math::
        y(t) = k t^{m}

    where :math:`k, m` are fitting parameters describing the liquid absorption
    rate and linearity of the liquid absorption process, respectively.

    Parameters
    ----------
    t : numpy.ndarray
        The time points to fit the power-law to.
    k : float
        The liquid absorption rate.
    m : float
        The linearity of the absorption process.

    Returns
    -------
     : numpy.ndarray
        The power-law results for the liquid absorption data.
    """
    return k * t**m


def schott(t: npt.NDArray, A: float, siginf: float, tr: float) -> npt.NDArray:
    """
    The Schott model to be fitted to the swelling profile following the equation

    .. math::
        \sigma(t) = \frac{t - t_{r}}{A + \frac{t - t_{r}}{\sigma_{\infty}}}

    where :math:`A, \sigma_{\infty}, t_{r}` describe the multiplicative
    reciprocal of the initial swelling rate :math:`\sigma (t_{s,i})`, the
    swelling rate after infinite time and the time delay before swelling becomes
    uniform, respectively.

    Parameters
    ----------
    t : numpy.ndarray
        The time points to fit the power-law to.
    A : float
        The multiplicative reciprocal of the initial swelling rate,
        :math:`\sigma(t_{s,i})`.
    siginf : float
        The swelling rate at infinite time.
    tr : float
        The time delay before swelling becomes uniform.

    Returns
    -------
     : numpy.ndarray
        The Schott model results for the swelling data.
    """

    return (t - tr) / (A + (t - tr) / siginf)

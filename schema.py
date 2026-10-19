from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

@dataclass
class Results:
    sas: npt.NDArray[np.float32]
    sas_uncert: npt.NDArray[np.float32]
    casL: npt.NDArray[np.float32]
    casR: npt.NDArray[np.float32]
    casAvg: npt.NDArray[np.float32]
    t_ai: float
    t_af: float
    t_si: float
    k: float
    m: float
    A: float
    siginf: float
    tr: float
    t_ai_idx: int
    t_af_idx: int
    t_si_idx: int
    k_err: float
    m_err: float
    A_err: float
    siginf_err: float
    tr_err: float
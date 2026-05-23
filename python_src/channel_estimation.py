import numpy as np
from .config import PILOT_VALUE

class PilotAidedEstimator:
    """Pilot-based channel estimation for block fading channels."""
    
    def __init__(self, pilot_interval: int = 100):
        self.pilot_interval = pilot_interval
    
    def estimate(self, rx_signal: np.ndarray, pilot_indices: np.ndarray) -> np.ndarray:
        """Least squares channel estimation from pilot symbols."""
        # Estimate channel from pilots
        h_pilots = rx_signal[pilot_indices] / PILOT_VALUE
        h_avg = np.mean(h_pilots)
        
        # Block fading: same channel for entire block
        return np.full(len(rx_signal), h_avg, dtype=np.complex64)

def equalize_signal(rx: np.ndarray, h_est: np.ndarray, noise_var: float = 1.0) -> np.ndarray:
    """MMSE equalization. noise_var = 10^(-SNR_dB/10) for proper MMSE."""
    eps = 1e-9
    denom = np.abs(h_est)**2 + noise_var
    return (np.conj(h_est) / (denom + eps)) * rx
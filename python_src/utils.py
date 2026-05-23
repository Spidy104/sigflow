import numpy as np

def generate_qpsk(n_syms: int, seed: int = None) -> np.ndarray:
    """Generate QPSK symbols with reproducible seed."""
    rng = np.random.default_rng(seed)
    bits_i = rng.integers(0, 2, n_syms)
    bits_q = rng.integers(0, 2, n_syms)
    return (2 * bits_i - 1) + 1j * (2 * bits_q - 1)

def hard_decision_qpsk(rx: np.ndarray) -> np.ndarray:
    """Map received samples to nearest QPSK constellation."""
    real = np.where(rx.real >= 0, 1, -1)
    imag = np.where(rx.imag >= 0, 1, -1)
    return real + 1j * imag

def compute_ber(tx: np.ndarray, rx: np.ndarray) -> float:
    """Compute Bit Error Rate for QPSK."""
    tx_sym = hard_decision_qpsk(tx)
    rx_sym = hard_decision_qpsk(rx)
    return np.mean(tx_sym != rx_sym)

def insert_pilots(data_symbols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Insert pilot symbols at regular intervals.
    
    Inserts a pilot at index 0, then every PILOT_INTERVAL samples thereafter.
    Returns the combined signal and the pilot indices.
    """
    from .config import PILOT_INTERVAL, PILOT_VALUE
    n_data = len(data_symbols)
    # Number of pilots: one at position 0, then every PILOT_INTERVAL
    n_pilots = (n_data + PILOT_INTERVAL - 1) // PILOT_INTERVAL + 1
    n_total = n_data + n_pilots
    
    tx = np.zeros(n_total, dtype=np.complex64)
    pilot_indices = []
    data_idx = 0
    
    for i in range(n_total):
        # Insert pilot at positions 0, PILOT_INTERVAL, 2*PILOT_INTERVAL, etc.
        if i % (PILOT_INTERVAL + 1) == 0:
            tx[i] = PILOT_VALUE
            pilot_indices.append(i)
        else:
            if data_idx < n_data:
                tx[i] = data_symbols[data_idx]
                data_idx += 1
    
    return tx, np.array(pilot_indices)

def extract_data_symbols(tx_with_pilots: np.ndarray, pilot_indices: np.ndarray) -> np.ndarray:
    """Remove pilot symbols to get clean data for BER."""
    data_indices = np.setdiff1d(np.arange(len(tx_with_pilots)), pilot_indices)
    return tx_with_pilots[data_indices], data_indices
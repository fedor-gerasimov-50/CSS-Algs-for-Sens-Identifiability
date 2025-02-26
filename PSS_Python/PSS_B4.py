import numpy as np
from scipy.linalg import qr, svd
from PSS_B1 import success_check

def PSS_B4(dydq, eta=1e-8, k=None):
    """
    Column Subset Selection Algorithm with PCA B4
    
    Args:
        dydq: Sensitivity matrix (n x p with n >= p)
        eta: Threshold for info matrix rank (default: 1e-8)
        k: Rank, or number of identifiable params
        
    Returns:
        UnId: Unidentifiable parameter indices
        Id: Identifiable parameter indices
        c: Success criteria for rank revealing factorization
    """
    # Initialize values
    UnId = []
    p = dydq.shape[1]
    Id = np.arange(p)
    c = []
    
    # Initial QR decomposition
    Q, R = qr(dydq, mode='economic')
    _, sing_vals, V = svd(R, full_matrices=False)
    
    if k is None:
        # If eta provided: find k
        sing_vals_diag = np.diag(sing_vals) if len(sing_vals.shape) > 1 else sing_vals
        ind = np.where((sing_vals_diag / sing_vals_diag[0]) > eta)[0]
        k = len(ind)
    
    if k > 0:
        # Compute largest singular vector v_1
        v_1 = V[:, 0]
        
        # Move magnitude largest element to the top
        max_ind = np.argmax(np.abs(v_1))
        P_tild = np.arange(p)
        P_tild[0], P_tild[max_ind] = P_tild[max_ind], P_tild[0]
        
        # Compute QR of R*P_tild
        B = R[:, P_tild]
        Q_tild, R_tild = qr(B, mode='economic')
        
        Q = Q @ Q_tild
        R = R_tild
        P = P_tild.copy()
        
        # Main loop
        for l in range(1, k):  # l starts from 1 to k-1
            R_22 = R[l:, l:]
            
            _, _, V_l = svd(R_22, full_matrices=False)
            v_l = V_l[:, 0]
            
            max_ind = np.argmax(np.abs(v_l))
            
            P_tild_loop = np.arange(len(v_l))
            P_tild_loop[0], P_tild_loop[max_ind] = P_tild_loop[max_ind], P_tild_loop[0]
            
            # Compute QR decomposition of R_22 * P_tild
            Q_tild_loop, R_tild_loop = qr(R_22[:, P_tild_loop], mode='economic')
            
            temp = P_tild_loop + l  # Shift by how many you've done
            temp_ind = np.concatenate([np.arange(l), temp])
            
            # Update everything
            Q_block = np.block([[np.eye(l), np.zeros((l, Q_tild_loop.shape[1]))],
                              [np.zeros((Q_tild_loop.shape[0], l)), Q_tild_loop]])
            Q = Q @ Q_block
            P[:] = P[temp_ind]
            R[l:, l:] = R_tild_loop
        
        Id = P[:k]
        UnId = P[k:]
        abs_err, rel_err, cond_S, cond_S1 = success_check(dydq, P, k, sing_vals)
        c = np.array([[abs_err[0], rel_err[0]],
                     [abs_err[1], rel_err[1]],
                     [cond_S, cond_S1]])
    
    return UnId, Id, c
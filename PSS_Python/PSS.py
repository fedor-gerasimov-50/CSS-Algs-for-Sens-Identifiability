import os, sys
import numpy as np
from scipy.linalg import qr, svd, block_diag

def CSS_quality_check(S, P, k, sing_vals=None):
    """
    Compute success metrics based on the reordered sensitivity matrix.
    
    Inputs:
      S     : sensitivity matrix (n x p)
      P        : permutation vector (0-indexed) for columns
      k        : number of identifiable parameters
      sing_vals: singular values (unused in this implementation but passed for interface compatibility)
      
    Outputs:
      abs_err  : absolute error (2-norm of residual)
      rel_err  : relative error (abs_err divided by ||S||_2)
      cond_S   : condition number of S
      cond_S1  : condition number of the identifiable parameter submatrix S1
    """
    S1 = S[:, P[:k]]
    
    # Compute the projection residual S - S1 S1^† S
    S1_pinv = np.linalg.pinv(S1)
    residual = S - S1 @ S1_pinv @ S
    abs_err = np.linalg.norm(residual, 2)
    rel_err = abs_err / np.linalg.norm(S, 2)
    cond_S = np.linalg.cond(S)
    cond_S1 = np.linalg.cond(S1)
    return abs_err, rel_err, cond_S, cond_S1

def Alg_4_1_PCA_B1(S, k, rtol=1e-8):
    """
    Select a permutation of the p columns of S (n x p) so that, after permutation,
    the sensitivity matrix S* has the following block structure (through an unpivoted QR):
    ```
         S* = S @ P = [ S1  S2 ]
         Q, R = qr(S*),   where R = [ R11  R12 ]
                                  [  0   R22 ]
    ```
    where sigma_{k+1}(S) <= ||R22||_2 <= 2^{p - k - 1}sigma_{k+1}(S).
    
    Parameters:
      S : ndarray, shape (n, p)
          The sensitivity matrix (Jacobian).
      k : int, 1 <= k < p
          The number of “identifiable” columns to keep in the first block.
    
    Returns:
      P : ndarray, shape (p,)
          An array of column indices (a permutation of range(p)).
      Q : ndarray, shape (n, p)
          The orthogonal factor from the final unpivoted QR of S[:, P].
      R : ndarray, shape (p, p)
          The final upper–triangular factor.
    
    This iterative reordering aims to “push” the column associated with the smallest singular value
    of the current block into the trailing position.
    """
    n, p = S.shape
    
    if k is None:
        singvals = svd(S, compute_uv=False)
        k = len(singvals[(singvals/singvals[0]) > rtol])
    
    if k <= 0 or k >= p:
        raise ValueError(f"{k = } does not satisfy 1 <= k < p")
        
    Q0, R0, P0 = qr(S, pivoting=True, mode="economic")
    P = np.array(P0)
    # print(P)

    Q, R = qr(S, mode="economic")
    
    for ell in range(p, k, -1):
        R11 = R[:ell, :ell]
        R12 = R[:ell, ell:]
        R22 = R[ell:, ell:]

        _, _, Vt = svd(R11)
        v_sigma_ell = Vt[-1, :]
        j = np.argmax(np.abs(v_sigma_ell))
        
        P_tilde = np.arange(ell)
        if j != ell - 1:
            P_tilde[j], P_tilde[ell - 1] = P_tilde[ell - 1], P_tilde[j]
        
        R11P_tilde = R11[:, P_tilde]
        Q_tilde, R11_tilde = qr(R11P_tilde, mode="economic")
        
        #  Update P, Q, and R
        Q = Q @ Q_tilde if p == ell else Q @ block_diag(Q_tilde, np.eye(p - ell))
        P[:ell] = P[P_tilde]
        R = R11_tilde if p == ell else block_diag(R11_tilde, R22)
        R[:ell, ell:] =  Q_tilde.T @ R12

    return P, Q, R

def Alg_4_2_PCA_B4(S, k, rtol=1e-8):
    """
    Select a permutation of the p columns of S (n x p) so that, after permutation, the sensitivity matrix S* has the following block structure (through an unpivoted QR):
    
         S* = S @ P = [ S1  S2 ]
         Q, R = qr(S*),   where R = [ R11  R12 ]
                                  [  0   R22 ]
    
    where 2^{-k + 1} sigma_{k}(S) <= sigma_{k}(R_11) = sigma_{k}(S_1) <= sigma_{k}(S).
    Parameters:
      S : ndarray, shape (n, p)
          The sensitivity matrix (Jacobian).
      k : int, 1 <= k < p
          The number of “identifiable” columns to keep in the first block.
      rtol : float, optional
          Relative tolerance for singular value computation.
          Default is 1e-8.
    
    Returns:
      P : ndarray, shape (p,)
          An array of column indices (a permutation of range(p)).
      Q : ndarray, shape (n, p)
          The orthogonal factor from the final unpivoted QR of S[:, P].
      R : ndarray, shape (p, p)
          The final upper–triangular factor.
    
    This iterative reordering aims to “push” the column associated with the smallest singular value
    of the current block into the trailing position.
    """
    n, p = S.shape
    
    if k is None:
        singvals = svd(S, compute_uv=False)
        k = len(singvals[(singvals/singvals[0]) > rtol])
    
    if k <= 0 or k >= p:
        raise ValueError(f"{k = } does not satisfy 1 <= k < p")
        
    P = np.arange(p)
    Q, R = qr(S, mode="economic")
    
    for ell in range(k):
        R11 = R[:ell, :ell]
        R12 = R[:ell, ell:]
        R22 = R[ell:, ell:]

        _, _, Vt = svd(R22)
        v_sigma1 = Vt[0, :]  # Corresponding to largest singular value
        j = np.argmax(np.abs(v_sigma1))
        
        P_tilde = np.arange(p - ell)
        if j != 0:
            P_tilde[j], P_tilde[0] = P_tilde[0], P_tilde[j]
        
        R22P_tilde = R22[:, P_tilde]
        Q_tilde, R22_tilde = qr(R22P_tilde, mode="economic")
        
        # Update P, Q, and R
        Q = Q @ Q_tilde if ell == 0 else Q @ block_diag(np.eye(ell), Q_tilde)
        P[ell:] = P[ell:][P_tilde]
        R = R22_tilde if ell == 0 else block_diag(R11, R22_tilde)
        R[:ell, ell:] = R12
        
        # print(f'It {ell+1} of k = {k}: P = {P}; P_tilde = {P_tilde}')

    return P, Q, R


def Alg_4_3_PCA_B3(S, k, rtol=1e-8):
    """
    Select a permutation of the p columns of S (n x p) so that, after permutation,
    the sensitivity matrix S* has the following block structure (through an unpivoted QR):
    
         S* = S @ P = [ S1  S2 ]
         Q, R = qr(S*),   where R = [ R11  R12 ]
                                  [  0   R22 ]
    
    Parameters:
      S : ndarray, shape (n, p)
          The sensitivity matrix (Jacobian).
      k : int, 1 <= k < p
          The number of “identifiable” columns to keep in the first block.
      rtol : float, optional
          Relative tolerance for singular value computation.
          Default is 1e-8.
    
    Returns:
      P : ndarray, shape (p,)
          An array of column indices (a permutation of range(p)).
      Q : ndarray, shape (n, p)
          The orthogonal factor from the final unpivoted QR of S[:, P].
      R : ndarray, shape (p, p)
          The final upper–triangular factor.
      
    """
    n, p = S.shape
    
    if k is None:
        singvals = svd(S, compute_uv=False)
        k = len(singvals[(singvals/singvals[0]) > rtol])
    
    if k <= 0 or k >= p:
        raise ValueError(f"{k = } does not satisfy 1 <= k < p")
        
    P = np.arange(p)
    Q, R = qr(S, mode="economic")
    
    for ell in range(k):
        R11 = R[:ell, :ell]
        R12 = R[:ell, ell:]
        R22 = R[ell:, ell:]

        U, sigma, Vt = svd(R22)
        # V = Vt.T
        # W = V[:, :k - ell].T
        W = Vt[:k - ell, :]
        
        col_norms = np.linalg.norm(W, axis=0)
        j_max = np.argmax(col_norms)
        
        P_tilde = np.arange(p - ell)
        if j_max != 0:
            P_tilde[0], P_tilde[j_max] = P_tilde[j_max], P_tilde[0]
        
        R22P_tilde = R22[:, P_tilde]
        Q_tilde, R22_tilde = qr(R22P_tilde, mode="economic")
        
        #  Update P, Q, and R
        Q = Q @ Q_tilde if ell == 0 else Q @ block_diag(np.eye(ell), Q_tilde)  # More efficient: Q[:, i:] = Q[:, i:] @ Qi
        P[ell:] = P[ell:][P_tilde]
        R = R22_tilde if ell == 0 else block_diag(R11, R22_tilde)
        R[:ell, ell:] = R12

    return P, Q, R


def pss_srrqr(S, k=None, rtol=1e-8, f=1.0):
    """
    Parameter Subset Selection using strong rank-revealing QR (Algorithm 4.4).
    
    This function partitions the parameter space of the sensitivity matrix S into identifiable (S1) and unidentifiable (S2) parameters.
    
    Inputs:
      S : numpy array (n x p) sensitivity matrix with n >= p.
      rtol  : (optional) threshold for the relative singular values; default 1e-8.
      k    : (optional) number of identifiable parameters if known; otherwise it is determined 
             by the threshold eta.
      f    : (optional) factor to scale the tolerance; default 1.0. # TODO: Note this has to be >1 in the paper.
             
    Outputs:
      UnId : indices (0-indexed) of the unidentifiable parameters.
      Id   : indices (0-indexed) of the identifiable parameters.
      c    : a 2x2 array containing success criteria:
               [ [abs_err, rel_err],
                 [cond_S, cond_S1] ]
    """
    n, p = S.shape
    assert n >= p
    
    if k is None:
        singvals = svd(S, compute_uv=False)
        k = len(singvals[(singvals/singvals[0]) > rtol])
    
    if k <= 0 or k >= p:
        raise ValueError(f"{k = } does not satisfy 1 <= k < p")
     
    Q, R, P = qr(S, pivoting=True)
    
    S_vals = np.linalg.svd(R, compute_uv=False)
    
    increase_found = True
    counter_perm = 0

    while increase_found:
        R11 = R[:k, :k]
        R12 = R[:k, k:]
        R11invR12 = np.linalg.solve(R11, R12)
        R22 = R[k:, k:]
        
        R11_inv = np.zeros((k, k))
        try: 
            R11_inv = np.linalg.inv(R11)
        except:
            print("R11 is not invertible, using pseudo inverse")
            R11_inv = np.linalg.pinv(R11)
        
        # Compute omega: the 2-norm of each row of R11_inv.
        omega = np.linalg.norm(R11_inv, axis=1)  # shape: (k,)
        
        # Compute gamma: the 2-norm of each column of R22.
        gamma = np.linalg.norm(R22, axis=0)      # shape: (p-k,)

        # Compute the matrix F = sqrt( (R11invR12)^2 + (outer(omega, gamma))^2) element-wise.
        F_matrix = np.sqrt(np.square(R11invR12) + np.square(np.outer(omega, gamma)))
        
        found = np.argwhere(F_matrix > f)
        if found.size == 0: 
            increase_found = False
        else:
            # Take the first such occurrence.
            i_idx, j_idx = found[0]

            col_i = i_idx
            col_j = j_idx + k
            
            R[:, [col_i, col_j]] = R[:, [col_j, col_i]]
            Q_tilde, R_tilde = qr(R, mode='economic')
            P[[col_i, col_j]] = P[[col_j, col_i]]
            Q = Q @ Q_tilde
            R = R_tilde
            counter_perm += 1

    # The first k columns in the permutation P correspond to the identifiable parameters.
    Id = P[:k]
    UnId = P[k:]
    # Compute success criteria
    abs_err, rel_err, cond_S, cond_S1 = CSS_quality_check(S, P, k, S_vals)
    c = np.array([[abs_err, rel_err], [cond_S, cond_S1]])

    return UnId, Id, c


if __name__ == "__main__":
    # print(os.path.dirname(os.path.realpath(__file__)))
    np.random.seed(0)
    n, p = 50, 10
    S = np.random.randn(n, 6) @ np.random.rand(6, p)
    k = 6
    
    # P, Q, R = np.arange(p), np.zeros((n, p)), np.zeros((p, p))
    
    css_algo_list = [Alg_4_1_PCA_B1, Alg_4_2_PCA_B4, Alg_4_3_PCA_B3, pss_srrqr]
    
    for i, css_algo in enumerate(css_algo_list):
        print('---------------------------------------------------')
        print(f'Function: {css_algo.__name__}:')
        if css_algo.__name__ != "pss_srrqr":
            P, Q, R = css_algo(S, k)
            abs_err, rel_err, cond_S, cond_S1 = CSS_quality_check(S, P, k)
            print(f'Identifiable parameters: {P[:k]}, Unidentifiable parameters: {P[k:]})')
            print(f'Abs err: {abs_err:.4e}, Rel err: {rel_err:.4e}, Cond S: {cond_S:.4e}, Cond S1: {cond_S1:.4e}')
        else:
            UnId, Id, c = css_algo(S, k)
            err_algo, cond_algo = c
            abs_err, rel_err = err_algo
            cond_S, cond_S1 = cond_algo
            print(f'Identifiable parameters: {Id}, Unidentifiable parameters: {UnId})')
            print(f'Abs err: {abs_err:.4e}, Rel err: {rel_err:.4e}, Cond S: {cond_S:.4e}, Cond S1: {cond_S1:.4e}')

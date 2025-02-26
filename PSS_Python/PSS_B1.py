import numpy as np
from scipy.linalg import qr, svd, block_diag

def success_check(S, P, k, sing_vals=None):
    """
    Determines algorithm performance criteria.
    
    Args:
        S: Sensitivity matrix (n x p)
        P: Permutation vector (1 x p)
        k: Numerical rank of S
        sing_vals: Optional; vector of singular values of S
        
    Returns:
        abs_err: Absolute errors in checking basis criteria 2.4 and 2.5
        rel_err: Relative errors in checking basis criteria 2.4 and 2.5
        cond_S: Condition number of host matrix
        cond_S1: Condition number of identifiable columns
    """
    abs_err = np.zeros(2)
    rel_err = np.zeros(2)
    
    # Determine if singular values provided to function already
    if sing_vals is None:
        sing_vals = svd(S, compute_uv=False)
    elif len(sing_vals.shape) > 1:  # If sing_vals is in matrix form
        sing_vals = np.diag(sing_vals)
    
    # Check criteria 2.4: |Sig(S)_k - Sig(S1)_k|
    S_perm = S[:, P]
    
    if k == len(P):
        S1 = S_perm
        cond_S = np.linalg.cond(S1)
        cond_S1 = cond_S
    else:
        S1 = S_perm[:, :k]
        S2 = S_perm[:, k:]
        
        sing_vals_S1 = svd(S1, compute_uv=False)
        sig_S1_k = sing_vals_S1[k-1]  # k-1 because Python is 0-indexed
        sig_k = sing_vals[k-1]
        
        cond_S = np.linalg.cond(S)
        cond_S1 = np.linalg.cond(S1)
        
        abs_err[0] = abs(sig_S1_k - sig_k)
        rel_err[0] = sig_S1_k / sig_k  # gamma_1
        
        # Check criteria 2.5: |||(I-S1*pinv(S1))*S|| - Sig(S)_{k+1}|
        sig_kp1 = sing_vals[k]
        X = np.linalg.lstsq(S1, S2, rcond=None)[0]
        crit_norm = np.linalg.norm(S2 - S1 @ X, ord=2)
        abs_err[1] = crit_norm
        rel_err[1] = abs_err[1] / sig_kp1  # gamma_2
    
    return abs_err, rel_err, cond_S, cond_S1

def algorithm_4_1(S, k):
    """
    Select a permutation of the p columns of S (n x p) so that, after permutation,
    the sensitivity matrix S* has the following block structure (through an unpivoted QR):
    
         S* = S @ P = [ S1  S2 ]
         Q, R = qr(S*),   where R = [ R11  R12 ]
                                  [  0   R22 ]
    
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
    
    The algorithm works as follows:
      1. Compute an initial QR factorization with pivoting of S. Let S* = S[:, P0] be S
         with columns permuted by the initial pivoting.
      2. Compute an unpivoted QR of S*.
      3. For ell from p downto k+1:
         (a) Consider the leading ell×ell block R11 of R.
         (b) Compute an SVD of R11.
         (c) Let v be the right singular vector corresponding to the smallest singular value.
         (d) Find the index (within [0, ell-1]) where |v| is maximized.
         (e) If that index is not ell–1, swap the corresponding columns in the leading ell block.
         (f) Update the permutation and recompute the QR on S*[:, :ell] to update R.
    
    This iterative reordering aims to “push” the column associated with the smallest singular value
    of the current block into the trailing position.
    """
    n, p = S.shape
    if not (1 <= k < p):
        raise ValueError("k must satisfy 1 <= k < p")
        
    Q0, R0, p0 = la.qr(S, pivoting=True, mode="economic")
    P = np.array(p0)  # current permutation (array of indices)
    
    # Make a copy of S with reordered columns
    S_perm = S[:, P].copy()
    
    # Step 2: Compute an unpivoted economic QR of S_perm.
    Q, R = qr(S_perm, mode="economic")
    
    for ell in range(p, k, -1):
        R11 = R[:ell, :ell]

        U, sigma, Vt = svd(R11)
        # The smallest singular value is sigma[-1] and its associated right singular vector is:
        v = Vt[-1, :]
        # Find the index where |v| is largest (this index is chosen to be “moved” to the bottom).
        j = np.argmax(np.abs(v))
        
        P_tilde = P.copy()
        if j != ell - 1:
            P_tilde[[j, ell - 1]] = P_tilde[[ell - 1, j]]
        
        Q_tilde, R11_tilde = qr(R11[:, [j, ell - 1]], mode="economic")
        
        Q = Q @ Q_tilde if p == ell else Q @ block_diag(Q_tilde, np.eye(p - ell))
        P = P @ P_tilde if p == ell else P @ block_diag(P_tilde, np.eye(p - ell))
        R[:ell, :ell] = R11_tilde   # Change upper left block
        R[ell:, :ell] = Q_tilde.T @ R[ell:, :ell]  # Change upper right block
        
        # if p == ell:
        #     Q = Q @ Q_tilde
        #     P[[j, ell - 1]] = P[[ell - 1, j]]  # Equivalent to P = P @ P_tilde
        #     R[:ell, :ell] = R11_tilde   # Change upper left block
        #     R[ell:, :ell] = Q_tilde.T @ R[ell:, :ell]  # Change upper right block
        # else:
        #     Q = Q @ block_diag(Q_tilde, np.eye(p - ell))
        #     P[[j, ell - 1]] = P[[ell - 1, j]]  # Equivalent to P = P @ block_diag(P_tilde, np.eye(p - ell))
        #     R[:ell, :ell] = R11_tilde
            
        
        # Recompute the QR factorization of the leading ell columns.
        Q_temp, R_temp = qr(S_perm[:, :ell], mode="economic")
        # Update the leading ell x ell block of R.
        R[:ell, :ell] = R_temp
        # (Optionally one might update the full S_perm[:, :ell] = Q_temp @ R_temp, but
        # since we only need to update R we reassemble the updated block.)
        
        
        # if j != ell - 1:
        #     # Swap columns j and ell-1 in the submatrix corresponding to the current block.
        #     # S_perm[:, [j, ell - 1]] = S_perm[:, [ell - 1, j]]
        #     # Update the permutation array accordingly.
        #     P[[j, ell - 1]] = P[[ell - 1, j]]
        #     Q_tilde, R11_tilde = qr(R11[:, [j, ell - 1]], mode="economic")
            
        #     # Recompute the QR factorization of the leading ell columns.
        #     Q_temp, R_temp = qr(S_perm[:, :ell], mode="economic")
        #     # Update the leading ell x ell block of R.
        #     R[:ell, :ell] = R_temp
        #     # (Optionally one might update the full S_perm[:, :ell] = Q_temp @ R_temp, but
        #     # since we only need to update R we reassemble the updated block.)
    # End for

    return P, Q, R

# Example of usage
if __name__ == "__main__":
    # Create a random sensitivity matrix S with n > p,
    # for example n=50, p=10.
    np.random.seed(0)
    n, p = 50, 10
    S = np.random.randn(n, p)
    
    # Let k be e.g. 6: we wish to select 6 “identifiable” columns.
    k = 6
    
    # Run Algorithm 4.1
    P, Q, R = algorithm_4_1(S, k)
    
    # Permutation P gives the new column ordering so that:
    S_perm = S[:, P]
    
    # Recover the partition: S_perm = [S1  S2] where S1 has k columns.
    S1 = S_perm[:, :k]
    S2 = S_perm[:, k:]
    
    print("Original column ordering:")
    print(np.arange(p))
    
    print("\nNew column ordering (permutation P):")
    print(P)
    
    print("\nShape of S1 (identifiable parameters):", S1.shape)
    print("Shape of S2 (unidentifiable parameters):", S2.shape)
    
    # Optionally, one may inspect the diagonal entries of R.
    diag_R = np.abs(np.diag(R))
    print("\nDiagonal of R (all p entries):")
    print(diag_R)
    
    # In an ideal case, the block R22 (diagonal entries from k to p-1) should be relatively small.


# def PSS_B1(dydq, eta=1e-8, k=None):
#     """
#     Column Subset Selection Algorithm with Alg 4.1 B1
    
#     Args:
#         dydq: Sensitivity matrix (n x p with n >= p)
#         eta: Threshold for info matrix rank (default: 1e-8)
#         k: Rank, or number of identifiable params
        
#     Returns:
#         UnId: Unidentifiable parameter indices
#         Id: Identifiable parameter indices
#         c: Success criteria for rank revealing factorization
#     """
#     # Initialize values
#     UnId = []
#     p = dydq.shape[1]
#     Id = np.arange(p)
#     c = []
    
#     # Initial QR decomposition
#     Q, R = qr(dydq, mode='economic')
#     _, sing_vals, V = svd(R, full_matrices=False)
    
#     if k is None:
#         # If eta provided: find k
#         sing_vals_diag = np.diag(sing_vals) if len(sing_vals.shape) > 1 else sing_vals
#         ind = np.where((sing_vals_diag / sing_vals_diag[0]) > eta)[0]
#         k = len(ind)
    
#     # Using k = rank(S1)
#     num_id = k
#     num_unid = p - num_id
    
#     if num_unid > 0:
#         # Compute smallest singular vector v_p
#         v_p = V[:, -1]
        
#         # Move magnitude largest element to the bottom
#         max_ind = np.argmax(np.abs(v_p))
#         P_tild = np.arange(p)
#         P_tild[max_ind], P_tild[-1] = P_tild[-1], P_tild[max_ind]
        
#         # Compute QR of R*P_tild
#         B = R[:, P_tild]
#         Q_tild, R_tild = qr(B, mode='economic')
        
#         Q = Q @ Q_tild
#         R = R_tild
#         P = P_tild.copy()
        
#         # Main loop
#         for iter in range(num_unid - 1):
#             l = p - iter - 1
#             R_11 = R[:l, :l]
#             R_12 = R[:l, l:]
#             R_22 = R[l:, l:]
            
#             _, _, V_l = svd(R_11, full_matrices=False)
#             v_l = V_l[:, -1]
            
#             max_ind = np.argmax(np.abs(v_l))
            
#             P_tild_loop = np.arange(len(v_l))
#             P_tild_loop[max_ind], P_tild_loop[-1] = P_tild_loop[-1], P_tild_loop[max_ind]
            
#             R_11P = R_11[:, P_tild_loop]
#             Q_tild_smlr, R_11_tild = qr(R_11P, mode='economic')
            
#             # Update Q, P, and R
#             Q_block = np.block([[Q_tild_smlr, np.zeros((Q_tild_smlr.shape[0], p-l))],
#                               [np.zeros((p-l, Q_tild_smlr.shape[1])), np.eye(p-l)]])
#             Q = Q @ Q_block
#             P[:l] = P[P_tild_loop]
#             R = np.block([[R_11_tild, Q_tild_smlr.T @ R_12],
#                          [np.zeros((p-l, l)), R_22]])
        
#         Id = P[:num_id]
#         UnId = P[num_id:]
#         sing_vals = sing_vals[sing_vals != 0] if len(sing_vals.shape) == 1 else np.diag(sing_vals)[np.diag(sing_vals) != 0]
#         abs_err, rel_err, cond_S, cond_S1 = success_check(dydq, P, num_id, sing_vals)
#         c = np.array([[abs_err[0], rel_err[0]],
#                      [abs_err[1], rel_err[1]],
#                      [cond_S, cond_S1]])
    
#     return UnId, Id, c
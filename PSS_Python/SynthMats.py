import numpy as np
from tqdm import tqdm, trange
from scipy.linalg import qr, svd, block_diag, null_space
from numpy.linalg import norm, cond, matrix_rank
# np.set_printoptions(precision=4, formatter={'float': '{:.4e}'.format})

def generate_gu_eisenstat_matrix(n=100, k=None, zeta_range=(0.9, 0.99999)):
    """
    Args:
        n: matrix dimensions (default: 90)
        k: numerical rank (default: n-2)
        zeta_range: tuple of (lower, upper) bounds for zeta draws
        
    Returns:
        M: Gu-Eisenstat matrix
        k: numerical rank
    """
    assert n >= 3
    
    if k is None:
        k = n - 2
        
    a_zeta, b_zeta = zeta_range
    
    # Sample zeta and compute phi
    zeta = a_zeta + (b_zeta - a_zeta) * np.random.rand()
    phi = np.sqrt(1 - zeta**2)
    
    # Construct matrices K and S
    K = np.eye(k-1)
    S = np.zeros((k-1, k-1))
    
    for row in range(k-1):
        S[row, row] = zeta**(row)
        for col in range(row+1, k-1):
            K[row, col] = -phi
    
    X = S @ K
    
    omega = np.zeros(k-1)
    Xinv = np.linalg.inv(X)
    for row in range(k-1):
        omega[row] = 1 / np.linalg.norm(Xinv[row, :], ord=2)
    
    mu = 1 / np.sqrt(k) * np.min(omega)
    T = -phi * S @ np.ones((k-1, 1))
    
    # Build final matrix M
    M = np.zeros((n, n))
    M[:k-1, :k-1] = X
    M[:k-1, n-1] = T.flatten()
    for row in range(k-1, n):
        M[row, row] = mu
    
    return M, k

def generate_kahan_matrix(n=100, k=None, zeta_range=(0.9, 0.99999)):
    """
    Args:
        n: dimensions n x n (default: 100)
        k: numerical rank (default: n-1)
        zeta_range: tuple of (lower, upper) bounds for zeta draws
        
    Returns:
        M: Kahan matrix
        k: numerical rank
    """
    if k is None:
        k = n - 1
    
    a_zeta, b_zeta = zeta_range
    
    # Sample zeta and compute phi
    zeta = a_zeta + (b_zeta - a_zeta) * np.random.rand()
    phi = np.sqrt(1 - zeta**2)
    
    # Construct factors for product S*K
    K = np.eye(n)
    S = np.zeros((n, n))
    
    for row in range(n):
        S[row, row] = zeta**(row)
        for col in range(row+1, n):
            K[row, col] = -phi
    
    M = S @ K
    
    return M, k

def generate_jolliffe_matrix(p=20, blksize=5, m=200, k=None, rho_range=(0.9, 0.99999)):
    """
    Args:
        p: number of blocks (default: 20)
        blksize: size of each block (default: 5)
        m: number of rows (default: 200)
        k: numerical rank (default: p)
        rho_range: tuple of (lower, upper) bounds for rho values
        
    Returns:
        M: Jolliffe matrix
        k: numerical rank
    """
    if k is None:
        k = p
    
    rho_1, rho_2 = rho_range
    n = blksize * p
    
    # Generate rho values
    rho = rho_1 + (rho_2 - rho_1) * np.random.rand(p)
    
    # Create Lambda blocks
    Lambda_blocks = []
    for block in range(p):
        lambda_block = rho[block] * np.ones((blksize, blksize))
        np.fill_diagonal(lambda_block, 1)
        Lambda_blocks.append(lambda_block)
    
    # Create block diagonal matrix
    LambdaMat = block_diag(*Lambda_blocks)
    
    # QR factorization of LambdaMat for V
    V, _ = qr(LambdaMat)
    
    # Generate U
    X = np.random.randn(m, n)
    U, _ = qr(X, mode='economic')
    
    # Generate singular values
    singvals = np.random.rand(n)
    
    # Scale singular values
    a1, b1 = 2, 3
    a2, b2 = -10, 1.9
    
    logscale1 = a1 + (b1 - a1) * np.random.rand(k)  # Not unfirormly, but uniformly on a log scale!!
    scale1 = 10 ** logscale1
    
    logscale2 = a2 + (b2 - a2) * np.random.rand(n - k)
    scale2 = 10 ** logscale2
    
    sing_vals1 = singvals[:k] * scale1  # k largest
    sing_vals2 = singvals[k:] * scale2  # n-k smallest
    
    sing_vals = np.concatenate([sing_vals1, sing_vals2])
    sing_vals = np.sort(sing_vals)[::-1]  # Sort in descending order
    
    S = np.diag(sing_vals)
    
    M = U @ S @ V.T
    
    return M, k

def generate_sorem_matrix(m=200, n=100, k=20):
    """
    Args:
        m: number of rows (default: 200)
        n: number of columns (default: 100)
        k: numerical rank (default: 20)
        
    Returns:
        M: Sorensen-Embree matrix
        k: numerical rank
    """
    # Create L matrix
    L_extended = -np.ones((n, k))
    L_extended[:k, :] = np.tril(-np.ones((k, k)), -1) + np.eye(k)
    
    # Construct V
    Vk, _ = qr(L_extended, mode='economic')
    Vp = null_space(Vk.T)
    V = np.hstack((Vk, Vp))
    
    # Generate U
    X = np.random.randn(m, n)
    U, _ = qr(X, mode='economic')
    
    singvals = np.random.rand(n)
    
    a1, b1 = 2, 3
    a2, b2 = -10, 1.9
    
    logscale1 = a1 + (b1 - a1) * np.random.rand(k)
    scale1 = 10 ** logscale1
    
    logscale2 = a2 + (b2 - a2) * np.random.rand(n - k)
    scale2 = 10 ** logscale2
    
    sing_vals1 = singvals[:k] * scale1  # k largest
    sing_vals2 = singvals[k:] * scale2  # n-k smallest
    
    sing_vals = np.concatenate([sing_vals1, sing_vals2])
    sing_vals = np.sort(sing_vals)[::-1]
    
    # Create diagonal matrix of singular values
    S = np.diag(sing_vals)
    
    # Form the final matrix
    M = U @ S @ V.T
    
    return M, k

def generate_ships_matrix(m=200, n=100, k=20):
    """
    Args:
        m: number of rows (default: 200)
        n: number of columns (default: 100)
        k: numerical rank (default: 20)
        
    Returns:
        M: SHIPS matrix
        k: numerical rank
    """
    # Create V11
    T = -np.triu(np.ones((k, k)), 1) + np.eye(k)
    V11 = T / (2 * np.linalg.norm(T, ord=2))
    
    # Cholesky decomposition
    R = np.linalg.cholesky(np.eye(k) - V11.T @ V11).T
    
    # Define singular values
    S1 = np.logspace(3, 2, k)  # k largest singular values
    S2 = np.logspace(1.9, -10, n - k)
    S_diag = np.concatenate([S1, S2])
    S = np.diag(S_diag)
    
    # Generate random orthogonal matrices
    Y = np.random.randn(m, n)
    U, _ = qr(Y, mode='economic')
    
    X = np.random.randn(n - k, k)
    Q, _ = qr(X, mode='economic')
    V21 = Q @ R
    
    # Form V
    Vk = np.vstack([V11, V21])
    Vp = null_space(Vk.T)
    V = np.hstack((Vk, Vp))
    
    M = U @ S @ V.T
    
    return M, k

def analyze_avg_cond(N_iter=1000):
    gu_eisenstat_results = []
    kahan_results = []
    jolliffe_results = []
    sorem_results = []
    ships_results = []
    
    for i in trange(N_iter):
        gu_eisenstat, k_eisenstat = generate_gu_eisenstat_matrix()
        kahan, k_kahan = generate_kahan_matrix()
        jolliffe, k_jolliffe = generate_jolliffe_matrix()
        sorem, k_sorem = generate_sorem_matrix()
        ships, k_ships = generate_ships_matrix()
        
        gu_eisenstat_results.append(cond(gu_eisenstat, 2))
        kahan_results.append(cond(kahan, 2))
        jolliffe_results.append(cond(jolliffe, 2))
        sorem_results.append(cond(sorem, 2))
        ships_results.append(cond(ships, 2))
        
    print(f"Average condition number: gu_eisenstat = {np.mean(gu_eisenstat_results):.4e}, kahan = {np.mean(kahan_results):.4e}, jolliffe = {np.mean(jolliffe_results):.4e}, sorem = {np.mean(sorem_results):.4e}, ships = {np.mean(ships_results):.4e}")
        

if __name__ == "__main__":
    print("Generating matrices...")
    matrix_algos = [generate_ships_matrix, generate_jolliffe_matrix, generate_sorem_matrix, generate_kahan_matrix, generate_gu_eisenstat_matrix]
    for i, alg in enumerate(matrix_algos):
        M, k = alg()
        print(f"Matrix by {alg.__name__}; Size = {M.shape}, rank = {k}, {matrix_rank(M) = }, Condition number: {cond(M, 2):.4e}")
    
    N_iter = 1000
    print(f"Analyzing average consition number over {N_iter} iterations...")
    analyze_avg_cond(N_iter)
    
    print("Done.")
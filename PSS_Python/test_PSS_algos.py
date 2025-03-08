# test_PSS_algos.py
# Tests the four PSS algorithms on simple example matrices
# This script generates several test matrices and applies all PSS algorithms
# to evaluate their performance

import numpy as np
from scipy.linalg import qr, svd, block_diag
import time
from PSS import Alg_4_1_PCA_B1, Alg_4_2_PCA_B4, Alg_4_3_PCA_B3, pss_srrqr, CSS_quality_check
np.set_printoptions(formatter={'float': '{:.4e}'.format})

def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "="*50)
    print(title)
    print("="*50)

def print_separator():
    """Print a separator line."""
    print("\n" + "-"*40)

def run_and_time_algorithm(algo_name, algo_func, matrix, k):
    """Run algorithm and time it."""
    print(f"\n--- Testing {algo_name} ---")
    
    start_time = time.time()
    
    if algo_name != "PSS srrqr (Alg 4.4)":
        P, Q, R = algo_func(matrix, k)
        abs_err, rel_err, cond_S, cond_S1 = CSS_quality_check(matrix, P, k)
    else:
        # The srrqr algorithm returns UnId, Id, c where c contains the metrics
        UnId, Id, c = algo_func(matrix, k)
        err_algo, cond_algo = c
        abs_err, rel_err = err_algo
        cond_S, cond_S1 = cond_algo
        P = np.concatenate((Id, UnId))  # Combine for consistent interface
    
    elapsed_time = time.time() - start_time
    
    # Print results
    print(f"Identifiable parameters: {P[:k]}")
    print(f"Unidentifiable parameters: {P[k:]}")
    print(f"Absolute error: {abs_err:.4e}")
    print(f"Relative error: {rel_err:.4e}")
    print(f"Condition number of S: {cond_S:.4e}")
    print(f"Condition number of S1: {cond_S1:.4e}")
    print(f"Execution time: {elapsed_time:.4e} seconds")
    
    return P, abs_err, rel_err, cond_S, cond_S1, elapsed_time

def main():
    print_section_header("Testing PSS Algorithms on Simple Example Matrices")
    
    # Define algorithms
    algorithms = [
        ("PSS B1 (Alg 4.1)", Alg_4_1_PCA_B1),
        ("PSS B4 (Alg 4.2)", Alg_4_2_PCA_B4),
        ("PSS B3 (Alg 4.3)", Alg_4_3_PCA_B3),
        ("PSS srrqr (Alg 4.4)", pss_srrqr)
    ]
    
    # Create test matrices
    print("Creating test matrices...")
    
    # Test Matrix 1: Simple rank-deficient matrix
    A1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k1 = 2  # Matrix has rank 2
    
    # Test Matrix 2: Identity with a small perturbation
    n = 5
    A2 = np.eye(n)
    A2[n-1, 0] = 1e-8
    k2 = n
    
    # Test Matrix 3: Block diagonal with different condition numbers
    B1 = np.array([[10, 1], [1, 10]])
    B2 = np.array([[1e-4, 0], [0, 1e-4]])
    A3 = block_diag(B1, B2)
    k3 = 4
    
    # Test Matrix 4: Hilbert matrix (well-known ill-conditioned matrix)
    n = 6
    A4 = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A4[i, j] = 1/(i+j+1)
    k4 = 4  # Choose a numerical rank
    
    # Test Matrix 5: Random matrix with rapid singular value decay
    m = 10
    n = 8
    k5 = 5
    U, _ = np.linalg.qr(np.random.randn(m, m))
    V, _ = np.linalg.qr(np.random.randn(n, n))
    s = np.concatenate([np.logspace(1, -4, k5), np.zeros(n-k5)])
    A5 = U[:, :n] @ np.diag(s) @ V.T
    
    # Store all test matrices in a list
    test_matrices = [A1, A2, A3, A4, A5]
    ranks = [k1, k2, k3, k4, k5]
    matrix_names = [
        'Rank-deficient (3x3)', 
        'Perturbed Identity', 
        'Block Diagonal', 
        'Hilbert', 
        'Random with SVD decay'
    ]
    
    # Display tolerance
    tol = 1e-8
    print(f"Using tolerance: {tol:.2e}\n")
    
    # Test each matrix with all algorithms
    for i, (matrix, k, name) in enumerate(zip(test_matrices, ranks, matrix_names)):
        print_section_header(f"Test Matrix {i+1}: {name} ({matrix.shape[0]}x{matrix.shape[1]})")
        
        # Display singular values
        s = svd(matrix, compute_uv=False)
        print("Singular values:", end=" ")
        for sv in s:
            print(f"{sv:.4e}", end=" ")
        print("\n")
        print(f"Using numerical rank k = {k}\n")
        
        # Test all algorithms
        for algo_name, algo_func in algorithms:
            try:
                run_and_time_algorithm(algo_name, algo_func, matrix, k)
            except Exception as e:
                # print(f"\n--- Testing {algo_name} ---")
                print(f"Error: {str(e)}")
    
    print("\nAll tests completed.")

if __name__ == "__main__":
    main()
import numpy as np
from tqdm import tqdm, trange
from scipy.linalg import qr, svd, block_diag, null_space
from numpy.linalg import norm, cond, matrix_rank
from PSS import CSS_quality_check, Alg_4_1_PCA_B1, Alg_4_2_PCA_B4, Alg_4_3_PCA_B3, pss_srrqr

from SynthMats import generate_gu_eisenstat_matrix, generate_kahan_matrix, generate_jolliffe_matrix, generate_sorem_matrix, generate_ships_matrix


# def analyze_matrices(N_iter=10000):
#     gu_eisenstat_results = []
#     kahan_results = []
#     jolliffe_results = []
#     sorem_results = []
#     ships_results = []
    
#     for i in range(N_iter):
#         gu_eisenstat = generate_gu_eisenstat_matrix()
#         kahan = generate_kahan_matrix()
#         jolliffe = generate_jolliffe_matrix()
#         sorem = generate_sorem_matrix()
#         ships = generate_ships_matrix()
    
#     # Analyze matrices
    
    
#     # Compute success metrics
#     gu_eisenstat_success = success_check(gu_eisenstat, gu_eisenstat_results[0], gu_eisenstat_results[1])
    
def PSS_apply_SynthMat(N_iter=10):
    matrix_func_list = [generate_gu_eisenstat_matrix, generate_kahan_matrix, generate_jolliffe_matrix, generate_sorem_matrix, generate_ships_matrix]
    algo_list = [Alg_4_1_PCA_B1, Alg_4_2_PCA_B4, Alg_4_3_PCA_B3, pss_srrqr]
    results = {}
    
    for i in trange(N_iter):
        for matrix_func in matrix_func_list:
            results[matrix_func.__name__] = []
            S, k = matrix_func()
            print(f'--------- PSS Quality on matrix by {matrix_func.__name__} with rank {k} ---------------------------')
            for css_algo in algo_list:
                print(f'{css_algo.__name__}:')
                if css_algo.__name__ != "pss_srrqr":
                    P, Q, R = css_algo(S, k)
                    abs_err, rel_err, cond_S, cond_S1 = CSS_quality_check(S, P, k)
                    print(f'Abs err: {abs_err:.4e}, Rel err: {rel_err:.4e}, Cond S: {cond_S:.4e}, Cond S1: {cond_S1:.4e}')
                    results[matrix_func.__name__].append((abs_err, rel_err, cond_S, cond_S1))
                else:
                    UnId, Id, c = css_algo(S, k)
                    err_algo, cond_algo = c
                    abs_err, rel_err = err_algo
                    cond_S, cond_S1 = cond_algo
                    print(f'Abs err: {abs_err:.4e}, Rel err: {rel_err:.4e}, Cond S: {cond_S:.4e}, Cond S1: {cond_S1:.4e}')
                    results[matrix_func.__name__].append((abs_err, rel_err, cond_S, cond_S1))

    return results
        
def PSS_simple_examples():
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f'--------- PSS Quality on matrix by  with rank 2 ----------------------------------------')
    P, Q, R = Alg_4_1_PCA_B1(A, 2)
    abs_err, rel_err, cond_S, cond_S1 = CSS_quality_check(A, P, 2)
    print(f'Abs err: {abs_err:.4e}, Rel err: {rel_err:.4e}, Cond S: {cond_S:.4e}, Cond S1: {cond_S1:.4e}')
    
    pass

if __name__ == "__main__":
    PSS_apply_SynthMat()
    print("Done.")
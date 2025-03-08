% test_PSS_algorithms.m
% Tests the four PSS algorithms (B1, B3, B4, and srrqr) on simple example matrices
% This script generates several test matrices and applies all PSS algorithms
% to evaluate their performance

% Clear workspace and command window
clear all;
clc;

fprintf('Testing PSS Algorithms on Simple Example Matrices\n');
fprintf('=================================================\n\n');

% Define test matrices
fprintf('Creating test matrices...\n');

% Test Matrix 1: Simple rank-deficient matrix
A1 = [1, 2, 3; 4, 5, 6; 7, 8, 9];
k1 = 2;  % Matrix has rank 2

% Test Matrix 2: Identity with a small perturbation
n = 5;
A2 = eye(n);
A2(n,1) = 1e-8;
k2 = n;

% Test Matrix 3: Block diagonal with different condition numbers
B1 = [10, 1; 1, 10];
B2 = [1e-4, 0; 0, 1e-4];
A3 = blkdiag(B1, B2);
k3 = 4;

% Test Matrix 4: Hilbert matrix (well-known ill-conditioned matrix)
n = 6;
A4 = zeros(n);
for i = 1:n
    for j = 1:n
        A4(i,j) = 1/(i+j-1);
    end
end
k4 = 4;  % Choose a numerical rank

% Test Matrix 5: Random matrix with rapid singular value decay
m = 10;
n = 8;
k5 = 5;
[U, ~] = qr(randn(m,m), 0);
[V, ~] = qr(randn(n,n), 0);
s = [logspace(1, -4, k5) zeros(1, n-k5)];
A5 = U(:,1:n) * diag(s) * V';

% Store all test matrices in a cell array
test_matrices = {A1, A2, A3, A4, A5};
ranks = [k1, k2, k3, k4, k5];
matrix_names = {'Rank-deficient (3x3)', 'Perturbed Identity', 'Block Diagonal', 'Hilbert', 'Random with SVD decay'};

% Create arrays for algorithm names and numbers
algo_names = {'PSS B1 (Alg 4.1)', 'PSS B4 (Alg 4.2)', 'PSS B3 (Alg 4.3)', 'PSS srrqr (Alg 4.4)'};
algo_nums = [1, 2, 3, 4];

% Display tolerance and test parameters
tol = 1e-8;
fprintf('Using tolerance: %.2e\n\n', tol);

% Test each matrix with all algorithms
for i = 1:length(test_matrices)
    A = test_matrices{i};
    k = ranks(i);
    
    fprintf('\n\n==========================================\n');
    fprintf('Test Matrix %d: %s (%dx%d)\n', i, matrix_names{i}, size(A,1), size(A,2));
    fprintf('==========================================\n');
    
    % Display singular values
    s = svd(A);
    fprintf('Singular values: ');
    fprintf('%.4e ', s);
    fprintf('\n');
    fprintf('Using numerical rank k = %d\n\n', k);
    
    % Test all algorithms
    for j = 1:length(algo_nums)
        algo_num = algo_nums(j);
        algo_name = algo_names{j};
        
        fprintf('\n--- Testing %s ---\n', algo_name);
        
        % Apply PSS algorithm
        tic;
        [unid, id, c] = PSS(A, tol, k, algo_num);
        % fprintf('Displaying c:\n')
        % disp(c)
        % disp(size(c))
        time_taken = toc;
        
        % Extract success criteria
        if ~isempty(c)
            err_metrics = c(1:2,:);
            abs_err = err_metrics(1,1);
            rel_err = err_metrics(1,2);
            cond_S = c(3,1);
            cond_S1 = c(3,2);
            
            % Display results
            fprintf('Identifiable parameters: ');
            fprintf('%d ', id);
            fprintf('\n');
            
            fprintf('Unidentifiable parameters: ');
            fprintf('%d ', unid);
            fprintf('\n');
            
            fprintf('Absolute error: %.4e\n', abs_err);
            fprintf('Relative error: %.4e\n', rel_err);
            fprintf('Condition number of S: %.4e\n', cond_S);
            fprintf('Condition number of S1: %.4e\n', cond_S1);
            fprintf('Execution time: %.4e seconds\n', time_taken);
        else
            fprintf('No unidentifiable parameters found.\n');
            fprintf('Execution time: %.4e seconds\n', time_taken);
        end
    end
end

fprintf('\n\nAll tests completed.\n');

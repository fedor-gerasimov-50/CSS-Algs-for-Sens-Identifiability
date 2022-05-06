% SVIR_main.m
% main file to run CSS algs on SVIR model using k or tol
% demonstrates how to generate sens matrix

% This code (1) demonstrates how to generate sensitivity matrix 
% using SVIR model in Section 2 of:
%   'Robust Parameter Identifiability Analysis via Column Subset Selection'
%       - Pearce, Ipsen, Haider, Saibaba, Smith
% and (2) demonstrates how to run CSS algs using either k or tol

clear 
close all

% Initialize
N = 332.6;
S0 = 295.1;
I0 = 1;
V0 = 0;
R0 = 0;

X0 = [S0; V0; I0; R0];

% Baseline Parameter Values
beta = 0.8;
alpha = 0.1; 
nu = 0.004; 
gamma = 0.14;

params = [beta; alpha; nu; gamma];

[t, y] = SVIR(X0, params);  % model
infct = y(:,3);             % infectious
infct = infct(1:10:end);    % in days

h = 1e-10; % step size for deriv approx

% Compute sensitivity matrix S_SVIR using complex-step deriv approximation
numpar = length(params);
numpts = length(infct);
S_SVIR = zeros(numpts,numpar);
for par = 1:numpar
    temp = params;
    temp(par) = complex(params(par),h); % complex perturbation

    [~,y_cs] = SVIR(X0,temp);
    y_cs = y_cs(:,3);
    y_cs = y_cs(1:10:end);

    S_SVIR(:,par) = imag(y_cs)/h;
end

PSS_method = 1:4; % all algorithms ([1] Alg 4.1; [2] Alg. 4.2; [3] Alg 4.3; [4] Alg 4.4)

%% passing in rank(S1)
% Unid: cell of indices of unidentifiable parameters for each algorithm
% crit: cell of criteria for each algorithm
k = 3;
[Unid_k, ~, crit_k] = SubsetSelect(S_SVIR, [], k, PSS_method);

%% passing in tolerance for singular value ratio
tol = 1e-3;
[Unid_tol, ~, crit_tol] = SubsetSelect(S_SVIR, tol, [], PSS_method);


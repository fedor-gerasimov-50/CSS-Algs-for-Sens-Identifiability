% filename: SHIPS.m

% requires: SubsetSelect.m, PSS.m, PSS_B1.m, PSS_B4.m, PSS_B3.m,
% PSS_srrqr.m, SuccessCheck.m

% Main file to construct SHIPS matrix realizations and perform CSS
% from 'Robust Parameter Identifiability Analysis' (Pearce et al. (2022))

clear 
close all

rng default 

Part = struct; % main structure holding relevant quantities for each realization

num_reals = 10000; % # realizations of SHIPS matrix
num_algs = 4;

m = 200; % # rows
n = 100; % # cols
k = 20; % num rank

V11 = -triu(ones(k,k),1) + eye(k);  % used to construct matrix of right singular vectors
V11 = V11/(2*norm(V11));
R = chol(eye(k) - V11'*V11, 'upper');

S1 = logspace(3,2,k)'; % k largest singular values
S2 = logspace(1.9,-10,n-k)'; 
S = diag([S1;S2]); % singular values

for real = 1:num_reals
    if mod(real,50) == 0
        sprintf('Iteration %d \n',real)
    end
    
    Y = randn(m,n); [U,~] = qr(Y,0);
    X = randn(n-k,k); [Q,~] = qr(X,0); V21 = Q*R;
    
    Vk = [V11; V21];
    
    Vp = null(Vk');
    V = [Vk, Vp]; % this will be the matrix of right singular vectors
   
    M = U*S*V';
    
    [UnId,~,crit] = SubsetSelect(M, [], k, 1:4);
    
    for alg = 1:num_algs
        Part(real).crit1{alg} = crit{alg}(1,1:2);
        Part(real).crit2{alg} = crit{alg}(2,1:2);
        Part(real).crit3{alg} = crit{alg}(3,2);
        Part(real).cond_num = crit{alg}(3,1);
    end

end

% extract information

crit1_abs = zeros(num_algs,num_reals); % criteria 2.4
crit1_rel = zeros(num_algs,num_reals); % gamma_1
crit2_abs = zeros(num_algs,num_reals); % criteria 2.5
crit2_rel = zeros(num_algs,num_reals); % gamma_2
crit3_cnd = zeros(num_algs,num_reals); % cond(S_1)/cond(S)

for real = 1:length(Part)
    for alg = 1:num_algs
        crit1_abs(alg,real) = Part(real).crit1{alg}(1,1);
        crit1_rel(alg,real) = Part(real).crit1{alg}(1,2);
        crit2_abs(alg,real) = Part(real).crit2{alg}(1,1);
        crit2_rel(alg,real) = Part(real).crit2{alg}(1,2);
        crit3_cnd(alg,real) = Part(real).crit3{alg}/Part(real).cond_num;
    end
end

% compute means for each algorithm
crit1_abs_mean = mean(crit1_abs,2);
crit1_rel_mean = mean(crit1_rel,2);
crit2_abs_mean = mean(crit2_abs,2);
crit2_rel_mean = mean(crit2_rel,2);
crit3_cnd_mean = mean(crit3_cnd,2);

cond_num = [Part.cond_num]';
mean_cond = mean(cond_num);

filename = sprintf('SHIPS_%d',num_reals);
save(filename)


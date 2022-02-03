%%% filename: GuEis.m

%%% requires: SubsetSelect.m, PSS.m, PSS_B1.m, PSS_B4.m, PSS_B3.m,
%%% PSS_srrqr.m, SuccessCheck.m

%%% constructs Gu-Eisenstat matrix realizations and performs column subset selection

close all
clear

rng default

Part = struct; %%% main structure to store relevant quantities from each realization

num_reals = 10000; %%% # realizations of Gu-Eis matrix
num_algs = 4;

n = 90; %%% matrix dimensions
k = n-2; %%% numerical rank

a_zeta = 0.9; %%% lower bound for zeta draws
b_zeta = 0.99999; %%% upper bound

for real = 1:num_reals
    if mod(real,25) == 0
        sprintf('Iteration %d \n',real)
    end

    zeta = a_zeta + (b_zeta-a_zeta)*rand; %%% sample zeta
    phi = sqrt(1-zeta^2);

    K = eye(k-1);
    S = zeros(k-1);
    
    for row = 1:k-1
        S(row,row) = zeta^(row-1);
        for col = row+1:k-1
            K(row, col) = -phi;
        end
    end

    X = S*K;
    omega = zeros(1,k-1); %%% store the row norms of inv(S_{k-1}*K_{k-1})
    Xinv = X\eye(k-1);
    for row = 1:k-1
        omega(row) = 1/norm(Xinv(row,:));
    end
    mu = 1/sqrt(k)*min(omega);
    c_vec = ones(k-1,1);
    T = -phi*S*c_vec;
    
    %% build M
    M = zeros(n);
    M(1:k-1,1:k-1) = X;
    M(1:k-1,n) = T;
    for row = k:n
        M(row,row) = mu;
    end

    [~,~,crit] = SubsetSelect(M, [], k, 1:4); %%% not storing UnId or Id in Part but you can below

    for alg = 1:num_algs
        Part(real).crit1{alg} = crit{alg}(1,1:2);
        Part(real).crit2{alg} = crit{alg}(2,1:2);
        Part(real).crit3{alg} = crit{alg}(3,2);
        Part(real).cond_num = crit{alg}(3,1);
    end
        
end

%%% extract information

crit1_abs = zeros(num_algs,num_reals); %%% criteria 2.4
crit1_rel = zeros(num_algs,num_reals); %%% gamma_1
crit2_abs = zeros(num_algs,num_reals); %%% criteria 2.5
crit2_rel = zeros(num_algs,num_reals); %%% gamma_2
crit3_cnd = zeros(num_algs,num_reals); %%% cond(S_1)/cond(S)

for real = 1:length(Part)
    for alg = 1:num_algs
        crit1_abs(alg,real) = Part(real).crit1{alg}(1,1);
        crit1_rel(alg,real) = Part(real).crit1{alg}(1,2);
        crit2_abs(alg,real) = Part(real).crit2{alg}(1,1);
        crit2_rel(alg,real) = Part(real).crit2{alg}(1,2);
        crit3_cnd(alg,real) = Part(real).crit3{alg}/Part(real).cond_num;
    end
end

%%% compute means for each algorithm
crit1_abs_mean = mean(crit1_abs,2);
crit1_rel_mean = mean(crit1_rel,2);
crit2_abs_mean = mean(crit2_abs,2);
crit2_rel_mean = mean(crit2_rel,2);
crit3_cnd_mean = mean(crit3_cnd,2);

cond_num = [Part.cond_num]';
mean_cond = mean(cond_num);

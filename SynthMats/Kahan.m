%%% filename: Kahan.m

%%% requires: SubsetSelect.m, PSS.m, PSS_B1.m, PSS_B4.m, PSS_B3.m,
%%% PSS_srrqr.m, SuccessCheck.m

%%% constructs Kahan matrix realizations and performs column subset selection

clear
close all

rng default 

Part = struct; %%% main structure to store relevant quantities from each realization

num_reals = 10000; %%% # realizations of Kahan matrix
num_algs = 4; 

n = 90; %%% dimensions n x n
k = n-1; %%% num rank

a_zeta = 0.9; %%% lower bound for zeta draws
b_zeta = 0.99999; %%% upper bound for zeta draws

for real = 1:num_reals
    if mod(real,25) == 0
        sprintf('Iteration %d \n',real)
    end

    zeta = a_zeta + (b_zeta-a_zeta)*rand; %%% sample zeta
    phi = sqrt(1-zeta^2);

    K = eye(n);
    S = zeros(n);
    
    for row = 1:n %%% construct factors for product S*K
        S(row,row) = zeta^(row-1);
        for col = row+1:n
            K(row, col) = -phi;
        end
    end

    [~,~,crit] = SubsetSelect(S*K, [], k, 1:4); %%% not storing UnId or Id but can below

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

%%% log10 of means
crit1_abs_mlog = log10(crit1_abs_mean);
crit1_rel_mlog = log10(crit1_rel_mean);
crit2_abs_mlog = log10(crit2_abs_mean);
crit2_rel_mlog = log10(crit2_rel_mean);
crit3_cnd_mlog = log10(crit3_cnd_mean);

clearvars -except Part crit1_abs crit1_rel crit2_abs crit2_rel crit3_cnd crit1_abs_mlog crit1_rel_mlog crit2_abs_mlog crit2_rel_mlog crit3_cnd_mlog

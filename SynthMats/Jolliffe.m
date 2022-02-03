%%% filename: Jolliffe.m

%%% requires: SubsetSelect.m, PSS.m, PSS_B1.m, PSS_B4.m, PSS_B3.m,
%%% PSS_srrqr.m, SuccessCheck.m

%%% constructs square Jolliffe matrix realizations and performs column subset selection

clear 
close all

rng default 

Part = struct;
num_reals = 10000;
num_algs = 4;

p = 20; %%% number of blocks

rho_1 = .9; %%% lower bound for rho
rho_2 = .99999; %%% upper bound

blksize = 5; %%% size of each block; can be different sizes, just change K
K = repmat(blksize,1,p); %%% vector of length p storing each block's size

n = blksize*p;
m = 200; 
k = p;

for real = 1:num_reals
    if mod(real,25) == 0
        sprintf('Iteration %d \n',real)
    end

    rho = rho_1 + (rho_2-rho_1).*rand(p,1); %%% values of rho_i
    
    Lambda = {};
    
    for block = 1:p
        lambda = ones(K(block));
        lambda = rho(block)*lambda;
        for rowcol = 1:K(block)
            lambda(rowcol,rowcol) = 1;
        end
        Lambda{block} = lambda;
    end
    
    LambdaMat = blkdiag(Lambda{:});

    %V = LambdaMat;
    [V,~] = qr(LambdaMat);

    X = randn(m,n);
    [U,~] = qr(X,0);

    singvals = rand(n,1);

    a1 = 2; b1 = 3;
    a2 = -10; b2 = 1.9;
    logscale1 = a1 + (b1-a1).*rand(k,1);
    scale1 = 10.^logscale1;
    logscale2 = a2 + (b2-a2).*rand(n-k,1);
    scale2 = 10.^logscale2;
    
    sing_vals1 = singvals(1:k).*scale1; %%% k largest
    sing_vals2 = singvals(k+1:n).*scale2; %%% n-k smallest
    
    sing_vals = sort([sing_vals1; sing_vals2],'descend');
    
    S = diag(sing_vals);
    
    M = U*S*V';
    [UnId,~,crit] = SubsetSelect(M, [], k, 1:4); 
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

filename = sprintf('Joll_%d.mat',num_reals);
save(filename)


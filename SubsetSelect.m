%%% filename: SubsetSelect.m

%%% Inputs: 
%%%     SensMat: sensitivity matrix (num data pts x num params)
%%%     tol: double, tolerance for num rank; default 1e-8
%%%     k: integer, numerical rank
%%%     algorithm: vector of PSS method options
%%%           0: PSS_eig (defunct)
%%%           1: PCA B1
%%%           2: PCA B4
%%%           3: PCA B3
%%%           4: srrqr

%%% Outputs:
%%%     UnId: cell of unidentifiable parameters for each method
%%%     Id: cell of identifiable parameters for each method (only need UnId or Id)
%%%     CondNum: condition number after unid cols removed

function [UnId, Id, criteria] = SubsetSelect(SensMat, tol, k, algorithm)

total_algs = 5;

num_algs = length(algorithm);

UnId = cell(1,total_algs);
Id = cell(1,total_algs);
criteria = cell(1,total_algs);

for alg=1:num_algs
    [unid, id, crit_vec] = PSS(SensMat, tol, k, algorithm(alg));
    
    UnId{1,algorithm(alg)+1} = unid; %%% for some reason I indexed algs starting at 0 like an absolute fool
    Id{1,algorithm(alg)+1} = id;
    criteria{1,algorithm(alg)+1} = crit_vec;

end



%%% filename: PSS.m

%%% Inputs: 
%%%     SensMat: sensitivity matrix (num data pts x num params)
%%%     tol: tolerance for num rank
%%%     algorithm: method option (pick 1)
%%%           0: PSS_eig
%%%           1: PCA B1
%%%           2: PCA B4
%%%           3: PCA B3
%%%           4: srrqr

%%% Outputs:
%%%     UnId: vector of unidentifiable parameters
%%%     Id: vector of identifiable parameters (only need UnId or Id)
%%%     c: vector of success criteria
%%%           first row: criteria 2.4 (abs and rel)
%%%           second row: criteria 2.5 (abs and rel)
%%%           third row: cond number of S and S1, resp

function [unid, id, c] = PSS(SensMat, tol, k, algorithm)

switch algorithm
    case 0
        [unid, id, c] = PSS_eig(SensMat, tol, k);
    case 1
        [unid, id, c] = PSS_B1(SensMat, tol, k);
    case 2
        [unid, id, c] = PSS_B4(SensMat, tol, k);
    case 3
        [unid, id, c] = PSS_B3(SensMat, tol, k);
    case 4
        [unid, id, c] = PSS_srrqr(SensMat, tol, k);
end


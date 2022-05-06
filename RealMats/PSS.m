% filename: PSS.m
% Applies specified CSS algorithm to sensitivity matrix

% Kate Pearce

function [unid, id, c] = PSS(SensMat, tol, k, algorithm)
% Inputs: 
%     SensMat: sensitivity matrix (num data pts x num params)
%     tol: tolerance for num rank
%     algorithm: method option 
%           1: Alg 4.1 (B1)
%           2: Alg 4.2 (B4)
%           3: Alg 4.3 (B3)
%           4: Alg 4.4 (srrqr)
% Outputs:
%     UnId: vector of unidentifiable parameters
%     Id: vector of identifiable parameters (set complement of UnId)
%     c: vector of success criteria
%           first row: criteria 3.4 (abs and rel)
%           second row: criteria 3.5 (abs and rel)
%           third row: cond number of S and S1, resp

switch algorithm
    case 1
        [unid, id, c] = PSS_B1(SensMat, tol, k);
    case 2
        [unid, id, c] = PSS_B4(SensMat, tol, k);
    case 3
        [unid, id, c] = PSS_B3(SensMat, tol, k);
    case 4
        [unid, id, c] = PSS_srrqr(SensMat, tol, k);
    otherwise 
        sprintf('Error: algorithm should be vector of integer(s) between 1:4')
end


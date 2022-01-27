%%% Parameter Subset Selection Algorithm with srrqr (PSS_srrqr.m)

%%% Determines unidentifiable parameters using parameter subset selection algorithm
%%% with strong rank-revealing QR factorization 

%%% Inputs:
    %%% eta: threshold for info matrix rank (double)
    %%% dydq: sensitivity matrix (num data pts by num params)
    
%%% Outputs:
    %%% UnId: unidentifiable parameter indices
    %%% Id: identifiable parameter indices
    %%% c: success criteria for rank revealing fact

function [UnId, Id, c] = PSS_srrqr(dydq, eta, k)

%%% initialize values
UnId = [];

p = size(dydq, 2);

Id = 1:p;
c = [];

%% do an initial QR decomposition: DON'T PIVOT FOR DIRECT ALG COMPARISONS
[Q, R] = qr(dydq, 0); 
[~, SingVals, ~] = svd(R, 'econ'); 

if (~exist('k', 'var')) || (isempty(k) == 1)
    if (~exist('eta', 'var')) || (isempty(eta) == 1) || (nargin < 2)
        eta = 1e-8; %% default
    end

    %% do the eta stuff: this will give a k
    SingVals = diag(SingVals);
    ind = find((SingVals./SingVals(1)) > eta); 
    k = length(ind); 

end

%% do the k stuff

if k > 0
    increasefound = true;
    counter_perm = 0;
    P = 1:p;

    while (increasefound)
        A = R(1:k,1:k); 
        AinvB = A\R(1:k,k+1:p); % Form A^{-1}B
        C = R(k+1:p, k+1:p);
        
        %compute the column norms of C
        gamma = zeros(p-k, 1);
        for ccol = 1:p-k
            gamma(ccol) = norm(C(:,ccol), 2);
        end
        
        %find row norms of A^-1
        [U, S, V] = svd(A);
        Ainv = V*diag(1./diag(S))*U';
        omega = zeros(k, 1);
        for arow = 1:k
            omega(arow) = norm(Ainv(arow,:), 2);
        end
        
        %find indices i and j that maximize 
        %ainv(i,j)^2 + (w(i)*gamma(j))^2
        tmp = omega*gamma';
        F = AinvB.^2 + tmp.^2;
        [i, j] = find(F>1, 1);
        if (isempty(i))           % finished
            increasefound = false;
        else  %we can increase |det(A)|
            counter_perm = counter_perm +1;
            R(:,[i j+k]) = R(:, [j+k i]);  % permute columns i and j
            P([i j+k]) = P([j+k i]);
            [Q, R] = qr(R, 0); % retriangularize R
        end   %if
    end   %while

    Id = P(1:k);  
    UnId = P(k+1:end);
    [abs_err, rel_err, cond_S, cond_S1] = SuccessCheck(dydq, P, k, SingVals);
    c = [abs_err, rel_err; cond_S, cond_S1] ;

end

% Parameter Subset Selection Algorithm with PCA B3 (PSS_B3.m)

% Determines unidentifiable parameters using 
% CSS Algorithm 4.3 from 'Robust Parameter Identifiability Analysis'
% (Pearce et al. 2022)

% Written by Kate Pearce

function [UnId, Id, c] = PSS_B3(dydq, eta, k)
% Inputs:
    % dydq: sensitivity matrix (double, n x p with n >= p)
    % eta (optional, double): threshold for info matrix rank; default 1e-8
    % k (optional, integer): rank, or number identifiable params   
% Outputs:
    % UnId: unidentifiable parameter indices
    % Id: identifiable parameter indices
    % c: success criteria for rank revealing fact

% initialize values
UnId = [];

p = size(dydq, 2);
Id = 1:p;

c = [];

% do an initial QR decomposition
[Q, R] = qr(dydq, 0); 
[~, SingVals, V] = svd(R, 'econ'); 

if (~exist('k', 'var')) || (isempty(k) == 1)
    if (~exist('eta', 'var')) || (isempty(eta) == 1) || (nargin < 2)
        eta = 1e-8; % default
    end

    % if eta provided: find k
    SingVals = diag(SingVals);
    ind = find((SingVals./SingVals(1)) > eta); 
    k = length(ind); 

end

% if k provided 

if k > 0
    W = V(:,1:k)';
    
    % move column w largest norm to front
    P_tild = 1:p;

    col_norms = zeros(1,p);
    for wcol = 1:p
        col_norms(wcol) = norm(W(:,wcol),2);
    end
    [~, max_ind] = max(col_norms);

    P_tild([1 max_ind]) = P_tild([max_ind 1]);
    
    % compute QR of R*P_tild
    B = R(:,P_tild);
    [Q_tild, R_tild] = qr(B);
    
    Q = Q*Q_tild;
    R = R_tild;
    P = P_tild;
    
    % main loop
    for l = 2:k % safe indexing: won't execute if k <= 1
        
        R_22 = R(l:end,l:end);
        
        [~,~,V_l] = svd(R_22);

        W = V_l(:,1:(k-l+1))';
        col_norms = zeros(1,p-l+1);
        for wcol = 1:(p-l+1)
            col_norms(wcol) = norm(W(:,wcol),2);
        end

        [~, max_ind] = max(col_norms);
        
        P_tild_loop = 1:(p-l+1);
        P_tild_loop([1 max_ind]) = P_tild_loop([max_ind 1]);
        
        % compute qr decomposition of R_22 * P_tild
        [Q_tild_loop, R_tild_loop] = qr(R_22(:,P_tild_loop));
        
        temp = P_tild_loop+ones(size(P_tild_loop))*(l-1); % shift for # iters done
        temp_ind = [1:(l-1),temp];

        % update
        Q = Q*blkdiag(eye(l-1), Q_tild_loop);
        P(1:p) = P(temp_ind);
        R(l:end,l:end) = R_tild_loop;

    end
    
    Id = P(1:k);  
    UnId = P(k+1:end);
    [abs_err, rel_err, cond_S, cond_S1] = SuccessCheck(dydq, P, k, SingVals); % criteria 
    c = [abs_err, rel_err; cond_S, cond_S1] ;
            
end

end
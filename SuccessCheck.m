% filename: SuccessCheck.m 

% Determines alg. performance criteria from `Robust Parameter
% Identifiability Analysis' (Pearce et al. (2022))

% Written by Kate Pearce

function [abs_err, rel_err, cond_S, cond_S1] = SuccessCheck(S, P, k, SingVals)
% Inputs: S (sensitivity matrix, n x p), 
%         P (permutation vector, 1 x p)
%         k (numerical rank of S)
%         SingVals (optional; vector, p x 1 singular values of S)
%         where submatrix column partition of S*P into [S1 S2] gives
%         approximate basis S1 for range(S) 

% Outputs: abs_err  (absolute errors in checking basis criteria 2.4 and 2.5)
%          rel_err  (relative errors in checking basis criteria 2.4 and 2.5)
%          cond_S   (condition number of host matrix)
%          cond_S1  (condition number of ident cols)   

   abs_err = zeros(2,1);
   rel_err = zeros(2,1);
   
   % determine if singular values provided to function already
   if ~exist('SingVals', 'var') || isempty(SingVals)
       SingVals = svd(S, 0);
   else
       if size(SingVals,2)~=1 % change to vec if SingVals in matrix form
           SingVals = diag(SingVals);
       end
   end
   
   % check criteria 2.4: |Sig(S)_k - Sig(S1)_k|
   S_perm = S(:,P);

   if k == length(P)
       S1 = S_perm;
       cond_S = cond(S1);
       cond_S1 = cond(S1);
       % fprintf('BIBA');

   else
       S1 = S_perm(:,1:k);
       S2 = S_perm(:,k+1:end);
       
       SingVals_S1   = svd(S1, 0);
       SigS1_k = SingVals_S1(k);
       Sig_k   = SingVals(k);
       
       cond_S = cond(S);
       % disp(S);
       % fprintf('Cond number: %.4f\n', cond_S);
       cond_S1 = cond(S1);
    
       abs_err(1) = abs(SigS1_k - Sig_k); 
       rel_err(1) = SigS1_k/Sig_k; % gamma_1
       
       % check criteria 2.5 | ||(I-S1*pinv(S1))*S|| - Sig(S)_{k+1}|
       Sig_kp1  = SingVals(k+1);
       X = S1\S2;
       crit_norm = norm(S2-S1*X, 2);
       abs_err(2) = crit_norm; 
       rel_err(2) = abs_err(2)/Sig_kp1; % gamma_2

   end
   
   
end
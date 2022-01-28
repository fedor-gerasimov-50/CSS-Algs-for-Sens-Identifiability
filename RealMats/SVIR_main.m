%%% SVIRmain.m

%%%% main file to run CSS algs on SVIR model using k
%%%% demonstrates how to generate sens matrix


clear 
close all


%%%%%%%%% Initialize

N = 332.6;
S0 = 295.1;
I0 = 1;
V0 = 0;
R0 = 0;

X0 = [S0; V0; I0; R0];

%%%%% Baseline Parameter Values
beta = 0.8;
alpha = 0.1; 
nu = 0.004; 
gamma = 0.14;

params = [beta; alpha; nu; gamma];

[t, y] = SVIR(X0, params);

h = 1e-10; %%% step size for deriv approx
numpar = length(params);
for par = 1:numpar
    temp = params;
    temp(par) = complex(params(par),h); %%% complex perturbation

    [~,y_cs] = SVIR(X0,temp);
    y_cs = y_cs(:,3);
    y_cs = y_cs(1:10:end,:);

    S_SVIR(:,par) = imag(y_cs)/h;
end

PSS_method = 1:4;
[Unid_SVIR, ~, crit_SVIR] = SubsetSelect(S_SVIR, [], 3, PSS_method);


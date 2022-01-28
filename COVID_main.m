%%% COVIDmain.m

%%%% main file to run CSS algs on COVID model
%%%% demonstrates how to generate sens matrix

clear 
close all

%%%%%% Initial Conditions
N = 332.6;

S0 = 285.1;
E0 = 10;
V0 = 0;
I0 = 1;
A0 = .3*I0;
H0 = .05*I0;
R0 = 0;

X0 = [S0; E0; V0; A0; I0; H0; R0];

%%%%%% Baseline Parameter Values
beta = 0.8;
alpha = 0.1;
eta = 0.33;
nu = 0.004;
gamma = 0.14;
sigma = .35; 
delta = 0.05; 
omega = 0.82;

params = [beta; alpha; nu; eta; gamma; sigma; delta; omega];

[t, y] = COVID(X0, params); 

h = 1e-10; %%% step size for complex-step deriv approx
numpar = length(params);

for par = 1:numpar
    temp = params;
    temp(par) = complex(params(par),h); %%% complex perturbation

    [~,y_cs] = COVID(X0,temp);
    y_cs = y_cs(:,4) + y_cs(:,5) + y_cs(:,6);
    y_cs = y_cs(1:10:end,:);

    S_COVID(:,par) = imag(y_cs)/h;
end

tol = 1e-4;
PSS_method = 1:4;
[Unid_COVID, ~, crit_COVID] = SubsetSelect(S_COVID, tol, [], PSS_method);


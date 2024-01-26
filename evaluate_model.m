%%% functions for constructing and evaluating mathematical models

function [time_vec,func_vec] = evaluate_model(model,init_cond,params)

arguments
    model string 
    init_cond (:,:) double = NaN
    params (:,:) double = NaN
end

switch model
    case 'COVID'
        % set initial conditions
        if isnan(init_cond) 
            X0 = [285.1; 10; 0; 1; 0];
        else 
            X0 = init_cond;   
        end
        % set parameter values
        if isnan(params)
            parameters = [0.8; 0.1; 0.004; 0.33; 0.14];
        else
            parameters = params;
        end
        % evaluate model
        [time_vec,func_vec] = covid_spread(X0,parameters);

    case 'OSN' % neuronal oscillation
        % set initial conditions
        if isnan(init_cond) 
            X0 = [-1.5;-3/8];
        else 
            X0 = init_cond;   
        end
        % set parameter values
        if isnan(params)
            parameters = [1.5; 1; 0.08; 10];
        else
            parameters = params;
        end
        [time_vec,func_vec] = single_neuronal_oscillation(X0,parameters);

    case 'SNB' % single neuronal bursting
        if isnan(init_cond) 
            X0 = [0.1; 1; 0.2];
        else 
            X0 = init_cond;   
        end
        % set parameter values
        if isnan(params)
            parameters = [1; 3; 1; 5; 0.005; 4; 0.4; -1.6];
        else
            parameters = params;
        end
        
        [time_vec,func_vec] = single_neuronal_bursting(X0, parameters);
    
    case 'Protein'
        [time_vec,func_vec] = protein_transduction(init_cond,params);
    case 'Ribosome'
        [time_vec,func_vec] = ribosome_control(init_cond,params);
    case 'Lorenz'
        [time_vec,func_vec] = chaotic_attractor(init_cond,params);
    case 'Vortex'
        [time_vec,func_vec] = vortex_shedding(init_cond,params);
end




%% COVID spread
%%%  Inputs: 
%%%       X0: initial condition for state vector X = [S; E; V; I; R]
%%%       params: model parameters
%%%           beta: transmission coeff = rate at which susceptible and infectious come into contact * probability of transmission [1/T * diml]
%%%           alpha: reduced fraction of transmission from vaccination (1 - vacc efficacy rate)
%%%           nu: vaccination rate
%%%           eta: rate of progression to infectious after exposure
%%%           gamma: rate of progression through infected stage   
%%% Outputs: 
%%%       t: time vector 
%%%       y: soln vector 
function [t,y] = covid_spread(X0, params)
    
    tfinal = 56; %%% Specify time steps to record the solution (in days)
    tspan = 0:0.1:tfinal; 
    
    N = sum(X0);
    
    options = odeset('AbsTol',1e-4); 
    
    frhs = @(t, y)(SEVIRrhs(t, y, params));    %%% anonymous sub-function for ODE solver
     
    [t,y] = ode45(frhs, tspan, X0, options); %%% calls numerical solver

    %%% sub-function SVIRrhs: creates right hand side of ODE system
    function yprime = SEVIRrhs(t, x, params)
               
        %%% state variables
        S = x(1);
        E = x(2);
        V = x(3);
        I = x(4);
        
        %%% model parameters
        beta = params(1);
        alpha = params(2);
        nu = params(3);
        eta = params(4);
        gamma = params(5);

        yprime = zeros(5, 1);

        yprime(1) = -beta * I * S/N - nu * S; %%% S
        yprime(2) = beta * I * S/N + alpha * beta * I * V/N - eta*E; %%% E
        yprime(3) = -alpha * beta * I * V/N + nu * S; %%% V
        yprime(4) = eta * E - gamma * I; %%% I
        yprime(5) = gamma * I; %%% R
    end

end



end

%% 

%% fitzhugh neuronal oscillation

function [t,y] = single_neuronal_oscillation(X0,params)
    N = 128;
    BC = 2;
    I = 6;
    tspan = 0:0.1:0.5;
    
    v0 = zeros(2*N,1);
    v0(1:N) = X0(1);
    v0(N+1:end) = X0(2);

    frhs = @(t, y)(OSNrhs(t, y, params));    %%% anonymous sub-function for ODE solver
     
    [t1,y1] = ode45(frhs, tspan, v0); %%% calls numerical solver
    
    tspan = 0.5:0.1:25;
    I = 0;
    [t2,y2] = ode45(frhs, tspan, y1(end,:)');

    t = [t1; t2(2:end)];
    y = [y1; y2(2:end,:)];

    %%% sub-function creates right hand side of ODE system
    function xdot = OSNrhs(t, x, params)
        xdot = zeros(2*N,1);
        a = params(1);
        b = params(2);
        p = params(3);
        D = params(4);

        xdot(1:N) = 10*(x(1:N) - x(1:N).^3/3 - x(N+1:end)) + D*secDer(x(1:N),1,BC)';
        xdot((round(N/2)-2):(round(N/2)+2)) = xdot((round(N/2)-2):(round(N/2)+2)) + I;
        xdot(N+1:end) = p*(1.25*x(1:N) + a - b*x(N+1:end));

        function V = secDer(v,dx,BC)
            F = [1 -2 1]./dx^2;
            switch BC
                case 1 % free boundary conditions
                    V = conv(v,F);
                    V = V(2:end-1);
                case 2 % periodic bc's
                    pv = zeros(1,length(v)+2);
                    pv(2:end-1) = v;
                    pv(1) = v(end);
                    pv(end) = v(1);
                    V = conv(pv,F);
                    V = V(3:end-2);
            end
        end

    end


end

%% Hindmarsh-Rose single neuronal bursting

function [t, y] = single_neuronal_bursting(X0,params)

tfinal =  256;
tspan = 0:0.1:tfinal;

frhs = @(t, y)(SNBrhs(t, y, params));    %%% anonymous sub-function for ODE solver

[t,y] = ode45(frhs, tspan, X0); %%% calls numerical solver

    %%% sub-function SNBrhs: creates right hand side of ODE system
    function yprime = SNBrhs(t, x, params)
               
        %%% state variables
        u = x(1);
        v = x(2);
        w = x(3);
        
        %%% model parameters
        a = params(1);
        b = params(2);
        c = params(3);
        d = params(4);
        r = params(5);
        s = params(6);
        I = params(7);
        u1 = params(8);

        yprime = zeros(3, 1);

        yprime(1) = v - a*u^3 + b*u^2 + I - w;  %%% u
        yprime(2) = c - d*u^2 - v; %%% v
        yprime(3) = r*(s*(u-u1) - w); %%% w
    end

end


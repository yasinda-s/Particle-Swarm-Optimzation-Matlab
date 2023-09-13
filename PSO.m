%% Source code referred from El-Abd, Mohammed & Kamel, Mohamed S. Kamel
%%(2009)

%% Reference - 
%%El-Abd, Mohammed & Kamel, Mohamed S. Kamel (2009). 
%%Black-box optimization benchmarking for noiseless function testbed using particle swarm optimization. 
%%2269-2274. 10.1145/1570256.1570316.

function PSO(FUN, DIM, ftarget, maxfunevals)
% MY_OPTIMIZER(FUN, DIM, ftarget, maxfunevals)
% samples new points uniformly randomly in [-5,5]^DIM
% and evaluates them on FUN until ftarget of maxfunevals
% is reached, or until 1e8 * DIM fevals are conducted. 
% Relies on FUN to keep track of the best point. 
    % Set algorithm parameters - initialization
    maxfunevals = min(1e8 * DIM, maxfunevals); %will take maxfunevalss
    popsize = min(maxfunevals, 200); %will take 200
    c1 = 1.5111;%2;
    c2 = 1.5111;%2;
    w = 0.792;
    xbound = 5;
    vbound = 5; 
    
    % Allocate memory and initialize
    xmin = -xbound * ones(1,DIM);
    xmax = xbound * ones(1,DIM);
    vmin = -vbound * ones(1,DIM);
    vmax = vbound * ones(1,DIM);
    
    %Position and velocity for initial iteration
    x = 2 * xbound * rand(popsize,DIM) - xbound;
    v = 2 * vbound * rand(popsize,DIM) - vbound;
    pbest = x; %pbest is whatever x is initially (randomly)

    % update pbest and gbest
    cost_p = feval(FUN, pbest');
    [cost,index] = min(cost_p);
    gbest = pbest(index,:); 
    
    maxiterations = ceil(maxfunevals/popsize); %25
    
    for iter = 2 : maxiterations
     % Update inertia weight
     w = 0.9 - 0.8*(iter-2)/(maxiterations-2);

     % Update velocity
     v = w*v + c1*rand(popsize,DIM).*(pbest-x) + c2*rand(popsize,DIM).*(repmat(gbest,popsize,1)-x);
    
    % Clamp veloctiy
     s = v < repmat(vmin,popsize,1);
     v = (1-s).*v + s.*repmat(vmin,popsize,1);
     b = v > repmat(vmax,popsize,1);
     v = (1-b).*v + b.*repmat(vmax,popsize,1);
     
     % Update position
     x = x + v;

     % Clamp position - Absorbing boundaries
     % Set x to the boundary
     s = x < repmat(xmin,popsize,1);
     x = (1-s).*x + s.*repmat(xmin,popsize,1);
     b = x > repmat(xmax,popsize,1);
     x = (1-b).*x + b.*repmat(xmax,popsize,1);
     
     % Clamp position - Absorbing boundaries 
     % Set v to zero
     b = s | b;
     v = (1-b).*v + b.*zeros(popsize,DIM);
     % Update pbest and gbest if necessary
     cost_x = feval(FUN, x');
     s = cost_x<cost_p;
     cost_p = (1-s).*cost_p + s.*cost_x;
     s = repmat(s',1,DIM);
     pbest = (1-s).*pbest + s.*x;
     [cost,index] = min(cost_p);
     gbest = pbest(index,:);

     % Exit if target is reached
     if feval(FUN, 'fbest') < ftarget
        break;
     end
    end
    

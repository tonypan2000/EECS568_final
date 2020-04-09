classdef liekf < handl
    
    properties
        mu;                 % Pose Mean
        Sigma;              % Pose Sigma
        gfun;               % Motion (process) model function
        mu_pred;            % Mean after prediction step
        Sigma_pred;         % Sigma after prediction step
        mu_cart;
        sigma_cart;
        M;                  % Motion model noise covariance function
        Q;                  % Measurement model noise covariance
    end
    
    methods
        function obj = liekf(init_mu, init_sigma)
            % liekf Construct an instance of this class
            %   Input: init          - motion and noise models
            obj.gfun = @(mu, u) ...
                [mu(1)+(-u(1)/u(2)*sin(mu(3))+u(1)/u(2)*sin(mu(3)+u(2)));
                mu(2)+(u(1)/u(2)*cos(mu(3))-u(1)/u(2)*cos(mu(3)+u(2)));
                mu(3)+u(2)+u(3)];
            obj.mu = init_mu;
            obj.Sigma = init_sigma;
            % Motion noise (in odometry space, Table 5.5, p.134 in book).
            % variance of noise proportional to alphas
            alphas = [0.00025 0.00005 0.0025 0.0005 0.0025 0.0005].^2; 
            obj.M = @(u) [alphas(1)*u(1)^2+alphas(2)*u(2)^2, 0, 0;
                        0, alphas(3)*u(1)^2+alphas(4)*u(2)^2, 0;
                        0, 0, alphas(5)*u(1)^2+alphas(6)*u(2)^2];
            % std. of Gaussian sensor noise (independent of distance)
            beta = deg2rad(5);
            obj.Q = [beta^2,    0;
                    0,      0.5^2];
        end
        
        function AdX = Ad(obj, X)
           % TODO:
           % Left-invariant Adjoint
           AdX = [X(1:2,1:2), [X(2,3); -X(1,3)]; 0 0 1];
        end
        
        function xhat = wedge(obj, x)
            % TODO
            % wedge operation for se(2) to put an R^3 vector into the Lie
            % algebra basis.
            G1 = [0    -1     0
                1     0     0
                0     0     0]; % omega
            G2 = [0     0     1
                0     0     0
                0     0     0]; % v_1
            G3 = [0     0     0
                0     0     1
                0     0     0]; % v_2
            xhat = G1 * x(1) + G2 * x(2) + G3 * x(3);
        end
        
        function H = posemat(state)
            x = state(1);
            y = state(2);
            z = state(3);
            roll = state(4);
            pitch = state(5);
            yaw = state(6);
            vx = state(7);
            vy = state(8);
            vz = state(9);
            % TODO: figure order of pitch and yaw
            euler_angle = [yaw pitch roll];
            rot_mat = eul2rotm(euler_angle);
            % construct a SE(3) matrix element
            H = [...
                rot_mat(1,1) rot_mat(1,2) rot_mat(1,3) vx x;
                rot_mat(2,1) rot_mat(2,2) rot_mat(2,3) vy y;
                rot_mat(3,1) rot_mat(3,2) rot_mat(3,3) vz z;
                0            0            0            1  0
                0            0            0            0  1];
        end
        
        function prediction(obj, u)
            % Formulate Adjoint function to be used in propagation
            % Convert motion command into lie algebra element to pass 
            % in to propagation
            % TODO: figure out order of roll and yaw
            [yaw, pitch, roll] = rotm2eul(obj.mu(1:3, 1:3));
            obj.mu_cart = [obj.mu(1,5); obj.mu(2,5); obj.mu(3,5); ...
                           roll; pitch; yaw; ...
                           obj.mu(1,4); obj.mu(2,4); obj.mu(3,4);];
            x_k1 = obj.gfun(obj.mu_cart, u);
            twist = logm(obj.mu \ posemat(obj, x_k1));

            % SE(2) propagation model; the input is u \in se(2) plus noise
            % propagate covariance
            obj.Sigma_pred = obj.Sigma + Ad(obj, obj.mu) * obj.M(u) * Ad(obj, obj.mu)';
            % propagate mean
            obj.mu_pred = obj.mu * expm(twist);
        end
        
        function correction(obj, Y1, b1, Y2, b2)
            % TODO
            % RI-EKF correction step
            H = [obj.H(b1); obj.H(b2)]; % stack H
            H = H([1:2,4:5],:); % 4x3 matrix, remove zero rows 
            N = obj.X * blkdiag(obj.N,0) * obj.X'; 
            N = blkdiag(N(1:2,1:2), N(1:2,1:2)); % 4x4 block-diagonal matrix
            % filter gain
            S = H * obj.P * H' + N;
            L = (obj.P * H') * (S \ eye(size(S)));
            
            % Update State
            nu = (blkdiag(obj.X, obj.X) * [Y1; Y2] - [b1; b2]); 
            nu([3,6]) = [];
            delta = obj.wedge( L * nu); % innovation in the spatial frame
            obj.X = expm(delta) * obj.X;
            
            % Update Covariance
            I = eye(size(obj.P));
            obj.P = (I - L * H) * obj.P * (I - L * H)' + L * N * L'; 
        end
   
    end
    
end
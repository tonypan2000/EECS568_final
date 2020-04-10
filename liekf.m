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
        dt_imu;             % IMU update period. TODO: check period, initialization
        g;                  % Gravity constant. TODO
    end
    
    methods
        function obj = liekf(init_mu, init_sigma)
            % liekf Construct an instance of this class
            %   Input: init          - motion and noise models
%             obj.gfun = @(mu, u) ...
%                 [mu(1)+(-u(1)/u(2)*sin(mu(3))+u(1)/u(2)*sin(mu(3)+u(2)));
%                 mu(2)+(u(1)/u(2)*cos(mu(3))-u(1)/u(2)*cos(mu(3)+u(2)));
%                 mu(3)+u(2)+u(3)];
            obj.gfun = obj.propagationModel;
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
                
            % IMU period initialization
            obj.dt_imu = 0.1;   %TODO
            
            % Gravity initialization
            obj.g = 9.81;
        end
        
        function AdX = Ad(obj, X)
           % TODO:
           % Left-invariant Adjoint
           AdX = [X(1:2,1:2), [X(2,3); -X(1,3)]; 0 0 1];
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
            
%             [yaw, pitch, roll] = rotm2eul(obj.mu(1:3, 1:3));
%             obj.mu_cart = [obj.mu(1,5); obj.mu(2,5); obj.mu(3,5); ...
%                            roll; pitch; yaw; ...
%                            obj.mu(1,4); obj.mu(2,4); obj.mu(3,4);];
%             x_k1 = obj.gfun(obj.mu_cart, u);

            % propagate covariance
            Phi = obj.Phi(u);
            obj.Sigma_pred = Phi*(obj.Sigma + obj.Q*obj.dt_imu)*Phi';
            % propagate mean
            obj.mu_pred = obj.gfun(u);
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
        
        function Xk1 = propagationModel(obj, u)
            % Status: completed
            % Assuming the structure of u to be the following:
            % u(1:3): IMU acceleration data ax, ay, az
            % u(4:6): IMU angular velocity data wx, wy, wz
            % Input: full IMU data
            % Output: X_{k+1}_pred \in SE_2(3)
            % The motion model is the one described in lecture slide
            % 05_invariant_ekf.pdf p.34
            
            H = obj.mu;
            R = H(1:3,1:3);
            v = H(1:3,4);
            p = H(1:3,5);
            R_pred = R*obj.Gamma0(u*obj.dt_imu);
            v_pred = v + R*obj.Gamma1(u*obj.dt_imu)*u(1:3)*obj.dt_imu + obj.g*obj.dt_imu;
            p_pred = p + v*obj.dt_imu +...
                R*obj.Gamma2(u*obj.dt_imu)*u(1:3)*obj.dt_imu^2 + 0.5*obj.g*obj.dt_imu^2;
            
            Xk1 = [R_pred, v_pred, p_pred;
                        0,      1,      0;
                        0,      0,      1 ];
        end
        
        function out = Gamma0(obj, u)
            % Status: complete
            % Please refer to lecture slide 05_invariant_ekf.pdf p.35
            phi = u(4:6);
            theta = norm(phi,2);
            out = eye(3) + sin(theta) / theta * obj.wedge_so3(phi) +...
                (1-cos(theta)) / theta^2 * (obj.wedge_so3(phi))^2;
        end
        
        function out = Gamma1(obj, u)
            % Status: complete
            % Please refer to lecture slide 05_invariant_ekf.pdf p.35
            phi = u(4:6);
            theta = norm(phi,2);
            out = eye(3) + (1-cos(theta)) / theta^2 * obj.wedge_so3(phi) +...
                (theta-sin(theta)) / theta^3 * (obj.wedge_so3(phi))^2;
        end
        
        function out = Gamma2(obj, u)
            % Status: complete
            % Please refer to lecture slide 05_invariant_ekf.pdf p.35
            phi = u(4:6);
            theta = norm(phi,2);
            out = 0.5*eye(3) + (theta - sin(theta)) / theta^3 * obj.wedge_so3(phi) +...
                (theta^2+2*cos(theta)-2) / (2*theta^4) * (obj.wedge_so3(phi))^2;
        end     
        
        function Phi_mat = Phi(obj, u)
            % Construct Phi matrix for covariance propagation
            % Please refer to lecture slide 05_invariant_ekf.pdf p.39
            % TODO: check wedge part in Phi21 and Phi31. not sure if
            % they're correct
            udt = u*obj.dt_imu;
            Phi11 = obj.Gamma0(udt)';
            Phi22 = Phi11;
            Phi33 = Phi11;
            Phi21 = -obj.Gamma0(udt)' * obj.wedge_so3(obj.Gamma1(udt)*u(1:3))*obj.dt_imu;
            Phi31 = -obj.Gamma0(udt)' * obj.wedge_so3(obj.Gamma2(udt)*u(1:3))*obj.dt_imu;
            Phi32 = obj.Gamma0(udt)'*obj.dt_imu;
            Phi_mat = [Phi11,   zeros(3),  zeros(3);
                       Phi21,   Phi22,     zeros(3);
                       Phi31,   Phi32,     Phi33    ];
        end
        
        function phi_wedge = wedge_so3(phi)
            % Status: complete
            % wedge operation for so(3)
            G1 = [ 0,  0,  0;
                   0,  0, -1;
                   0,  1,  0 ];
            G2 = [ 0,  0,  1;
                   0,  0,  0;
                  -1,  0,  0 ];
            G3 = [ 0, -1,  0;
                   1,  0,  0;
                   0,  0,  0 ];
            phi_wedge = phi(1)*G1 + phi(2)*G2 + phi(3)*G3;
        end
    end
    
end
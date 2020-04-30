classdef liekf < handle
    %LIEKF Left-Invariant EKF with IMU bias estimation.
    properties
        mu;                 % Pose Mean. 5x5
        Sigma;              % Pose Sigma. 15x15
        gfun;               % Motion (process) model function
        mu_pred;            % Mean after prediction step 5x5
        Sigma_pred;         % Sigma after prediction step 15x15
        mu_cart;            % Mean in Cartesian coordinate
                            % [roll pitch yaw vx vy vz px py pz] 9x1
        sigma_cart;         % Covariance in Cartesian coordinate
        M;                  % Motion model noise covariance function. 15x15
        Q;                  % Measurement model noise covariance. 3x3
        dt_imu;             % IMU update period. TODO: check period, initialization
        g;                  % Gravity vector. TODO
        b_a;                % IMU bias for accelerometer. 3x1
        b_g;                % IMU bias for gyroscope. 3x1
        theta_b;            % IMU bias vector. theta = [b_g; b_a] 6x1
        
        se3;                % state representation in SE3
        pose;               % odometry pose for visual processing
    end
    
    methods
        function obj = liekf(init_mu, init_sigma, init_imu_bias)
            %LIEKF Construct an instance of this class
            %   Input: init          - motion and noise models
            obj.gfun = @(u) obj.propagationModel(u);
            obj.mu = init_mu;
            obj.Sigma = init_sigma;
            obj.theta_b = init_imu_bias;
            % Motion noise (in odometry space, Table 5.5, p.134 in book).
            % variance of noise proportional to alphas
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            M = repmat(0.01^2, 15, 1);
            obj.M = diag(M);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % std. of Gaussian sensor noise (independent of distance)
            obj.Q = diag([0.01, 0.01, 0.01]);
                
            % IMU period initialization
            obj.dt_imu = 0.1;
            
            % Gravity initialization
            obj.g = [0, 0, -9.81]';
        end
        
        function X = posemat(state)
            x = state(1);
            y = state(2);
            z = state(3);
            roll = state(4);
            pitch = state(5);
            yaw = state(6);
            vx = state(7);
            vy = state(8);
            vz = state(9);
            euler_angle = [yaw pitch roll];
            rot_mat = eul2rotm(euler_angle);
            % construct a SE(3) matrix element
            X = [...
                rot_mat(1,1) rot_mat(1,2) rot_mat(1,3) vx x;
                rot_mat(2,1) rot_mat(2,2) rot_mat(2,3) vy y;
                rot_mat(3,1) rot_mat(3,2) rot_mat(3,3) vz z;
                0            0            0            1  0
                0            0            0            0  1];
        end
        
        function prediction(obj, u)
            % propagate mean
            obj.mu_pred = obj.gfun(u);
            
            % propagate covariance
            Phi = obj.Phi(u);
            obj.Sigma_pred = Phi*(obj.Sigma + obj.M*obj.dt_imu)*Phi';
        end
        
        function correction(obj, y)
            %CORRECTION Correction step of LIEKF.
            %   Input y should be 3x1.
            Y = [y; 0; 1];  %5x1
            R = obj.mu_pred(1:3,1:3);
            b = [zeros(3,1); 0; 1]; %5x1
            H = [zeros(3), zeros(3), eye(3), zeros(3), zeros(3)];   %3x15
            N = R' * obj.Q * R; %3x3
            S = H * obj.Sigma_pred * H' + N;    %3x3
            L = obj.Sigma_pred * H' / S; % Kalman gain 15x3
            % Covariance update
            I15 = eye(15);
            obj.Sigma = (I15-L*H) * obj.Sigma_pred * (I15-L*H)' + L*N*L';
            
            % Mean update
            nu = obj.mu_pred \ Y - b;
            PI = [eye(3), zeros(3,2)]; %3x5
            obj.mu = obj.mu_pred * expm(obj.wedge_se23(L(1:9,:)*PI*nu));
            obj.theta_b = obj.theta_b + L(10:15,:)*PI*nu;
            obj.lie2cart();
            
            obj.se3 = [obj.mu(1:3,1:3) obj.mu(1:3,5); 0 0 0 1];
            obj.pose = [obj.se3(1,1:4) obj.se3(2,1:4) obj.se3(3,1:4)];
        end
        
        function Xk1 = propagationModel(obj, u)
            % Status: completed
            % Assuming the structure of u to be the following:
            % u(1:3): IMU angular velocity data wx, wy, wz
            % u(4:6): IMU acceleration data ax, ay, az
            % Input: full IMU data
            % Output: X_{k+1}_pred \in SE_2(3)
            % The motion model is the one described in lecture slide
            % 05_invariant_ekf.pdf p.34
            
            u_unbias = u - obj.theta_b;
            omega_unbias_dt = u_unbias(1:3)*obj.dt_imu;
            accel_unbias = u_unbias(4:6);
            
            H = obj.mu;
            R = H(1:3,1:3);
            v = H(1:3,4);
            p = H(1:3,5);
            R_pred = R*obj.Gamma0(omega_unbias_dt);
            v_pred = v + R*obj.Gamma1(omega_unbias_dt)*accel_unbias*obj.dt_imu...
                + obj.g*obj.dt_imu;
            p_pred = p + v*obj.dt_imu + R*obj.Gamma2(omega_unbias_dt)...
                *accel_unbias*obj.dt_imu^2 + 0.5*obj.g*obj.dt_imu^2;
            
            Xk1 = [R_pred, v_pred, p_pred;
                   0, 0,0,      1,      0;
                    0,0,0,      0,      1 ];
        end
        
        function out = Gamma0(obj, phi)
            % Status: complete
            % Please refer to lecture slide 05_invariant_ekf.pdf p.35
            theta = norm(phi,2);
            out = eye(3) + sin(theta) / theta * obj.wedge_so3(phi) +...
                (1-cos(theta)) / theta^2 * (obj.wedge_so3(phi))^2;
        end
        
        function out = Gamma1(obj, phi)
            % Status: complete
            % Please refer to lecture slide 05_invariant_ekf.pdf p.35
            theta = norm(phi,2);
            out = eye(3) + (1-cos(theta)) / theta^2 * obj.wedge_so3(phi) +...
                (theta-sin(theta)) / theta^3 * (obj.wedge_so3(phi))^2;
        end
        
        function out = Gamma2(obj, phi)
            % Status: complete
            % Please refer to lecture slide 05_invariant_ekf.pdf p.35
            theta = norm(phi,2);
            out = 0.5*eye(3) + (theta - sin(theta)) / theta^3 * obj.wedge_so3(phi) +...
                (theta^2+2*cos(theta)-2) / (2*theta^4) * (obj.wedge_so3(phi))^2;
        end     
        
        function Phi_mat = Phi(obj, u)
            %PHI Construct Phi matrix for covariance propagation. This is
            %   specifically for the biased-version of Lef-Invariant 
            %   Extended Kalman Filter.
            %   Please refer to lecture slide 05_invariant_ekf.pdf p.36 and
            %   the paper "Contact-Aided Invariant Extended Kalman Filter
            %   for Legged Robots" p.28
            u_unbias = u - obj.theta_b;
            omega_unbias = [u_unbias(1:3)];
            accel_unbias = u_unbias(4:6);
            
            omega_wedge = obj.wedge_so3(omega_unbias);
            zero_3 = zeros(3);
            A_l = [-omega_wedge, zero_3, zero_3, -eye(3), zero_3;
                   -obj.wedge_so3(accel_unbias), -omega_wedge, zero_3, zero_3, -eye(3);
                   zero_3, eye(3), -omega_wedge, zero_3, zero_3;
                   zeros(6,15)                                                      ];
            Phi_mat = expm(A_l);
        end
        
        function phi_wedge = wedge_se23(obj,phi)
            %WEDGE_SE23 Construct the wedge matrix from vector in R^9
            %   to 5x5
            R = obj.wedge_so3(phi(1:3));
       
            phi_wedge = [R, phi(4:6), phi(7:9);
                         zeros(2,5)            ];
        end
        
        function phi_wedge = wedge_so3(obj,phi)
            %WEDGE_SO3 Construct the skew symmetric matrix from the
            %   correspoinding vector in R^3
            %   Status: complete
            %   wedge operation for so(3)
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
        
        function lie2cart(obj)
            % Status: complete
            % lie to cartesian transformation
            eul = rotm2eul(obj.mu(1:3,1:3));
            roll = eul(3);
            pitch = eul(2);
            yaw = eul(1);
            obj.mu_cart = [roll; pitch; yaw; obj.mu(1:3,4); obj.mu(1:3,5)];
        end
    end
    
end
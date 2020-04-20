classdef liekf < handle
    
    properties
        mu;                 % Pose Mean 5x5
        Sigma;              % Pose Sigma 9x9
        gfun;               % Motion (process) model function
        mu_pred;            % Mean after prediction step 5x5
        Sigma_pred;         % Sigma after prediction step 9x9
        mu_cart;            % Mean in Cartesian coordinate 9x1
                            % [roll, pitch, yaw, vx, vy, vz, px, py, pz]
        sigma_cart;
        M;                  % Motion model noise covariance 6x6
        Q;                  % Measurement model noise covariance 9x9
        dt_imu;             % IMU update period. TODO: check period, initialization
        g;                  % Gravity constant. TODO
    end
    
    methods
        function obj = liekf(init_mu, init_sigma)
            %LIEKF Construct an instance of this class
            obj.gfun = obj.propagationModel;
            obj.mu = init_mu;
            obj.Sigma = init_sigma;
            % Motion noise (in odometry space, Table 5.5, p.134 in book).
            % variance of noise proportional to alphas
            % TODO: check motion noise covariance dimension
            alphas = [0.00025 0.00005 0.0025 0.0005 0.0025 0.0005].^2; 
            obj.M = @(u) [alphas(1)*u(1)^2+alphas(2)*u(2)^2, 0, 0;
                        0, alphas(3)*u(1)^2+alphas(4)*u(2)^2, 0;
                        0, 0, alphas(5)*u(1)^2+alphas(6)*u(2)^2];
            % std. of Gaussian sensor noise (independent of distance)
            obj.Q = diag([0.01, 0.01, 0.01]);
                
            % IMU period initialization
            obj.dt_imu = 0.1;   %TODO
            
            % Gravity initialization
            obj.g = [0, 0, -9.81]';
        end
        
        function X = posemat(state)
            %POSEMAT Construct the state matrix in se2(3) from R^9.
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
            % construct a SE2(3) matrix element
            X = [...
                rot_mat(1,1) rot_mat(1,2) rot_mat(1,3) vx x;
                rot_mat(2,1) rot_mat(2,2) rot_mat(2,3) vy y;
                rot_mat(3,1) rot_mat(3,2) rot_mat(3,3) vz z;
                0            0            0            1  0
                0            0            0            0  1];
        end
        
        function prediction(obj, u)
            %PREDICTION Formulate Adjoint function to be used in propagation
            %   Convert motion command into lie algebra element to pass 
            %   in to propagation

            % propagate covariance
            Phi = obj.Phi(u);
            obj.Sigma_pred = Phi*(obj.Sigma + obj.M*obj.dt_imu)*Phi';
            
            % propagate mean
            obj.mu_pred = obj.gfun(u);
        end
        
        function correction(obj, y)
            
            Y = [y; 0; 1]; %5x1
            
            R = obj.mu_pred(1:3,1:3);
            b = [zeros(3,1); 0; 1];
            H = [zeros(3), zeros(3), eye(3)];
            N = R \ obj.Q / R'; % TODO: this makes no sense!
            S = H * obj.Sigma_pred * H' + N;
            L = obj.Sigma_pred*H' / S; % Kalman gain 9x3
            % Covariance update
            I9 = eye(9);
            obj.Sigma = (I9-L*H) * obj.Sigma_pred * (I9-L*H)' + L*N*L';
            % Mean update
            PI = [eye(3), zeros(3,2)]; %3x5
            nu = obj.mu_pred \ Y - b;  %5x1
            obj.mu = obj.mu_pred * expm(obj.wedge_se23(L*PI*nu)); %TODO: check Y, b
            obj.lie2cart();
        end
        
        function Xk1 = propagationModel(obj, u)
            %PROPAGATIONMODEL Propagates the mean and covariance using IMU
            %   measurements as input.
            %   u(1:3): IMU angular velocity data wx, wy, wz
            %   u(4:6): IMU acceleration data ax, ay, az
            %   Input: full IMU data
            %   Output: X_{k+1}_pred \in SE_2(3)
            %   The motion model is the one described in lecture slide
            %   05_invariant_ekf.pdf p.34
            
            omega_dt = u(1:3);
            accel = u(4:6);
            
            H = obj.mu;
            R = H(1:3,1:3);
            v = H(1:3,4);
            p = H(1:3,5);
            R_pred = R*obj.Gamma0(omega_dt);
            v_pred = v + R*obj.Gamma1(omega_dt)*accel*obj.dt_imu + obj.g*obj.dt_imu;
            p_pred = p + v*obj.dt_imu +...
                R*obj.Gamma2(omega_dt)*accel*obj.dt_imu^2 + 0.5*obj.g*obj.dt_imu^2;
            
            Xk1 = [R_pred, v_pred, p_pred;
                        0,      1,      0;
                        0,      0,      1 ];
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
            % Construct Phi matrix for covariance propagation
            % Please refer to lecture slide 05_invariant_ekf.pdf p.39
            omega_dt = u(1:3)*obj.dt_imu;
            accel = u(4:6);
            
            Phi11 = obj.Gamma0(omega_dt)';
            Phi22 = Phi11;
            Phi33 = Phi11;
            Phi21 = -obj.Gamma0(omega_dt)' * obj.wedge_so3(obj.Gamma1(omega_dt)*accel)...
                * obj.dt_imu;
            Phi31 = -obj.Gamma0(omega_dt)' * obj.wedge_so3(obj.Gamma2(omega_dt)*accel)...
                * obj.dt_imu^2;
            Phi32 = obj.Gamma0(omega_dt)'*obj.dt_imu;
            Phi_mat = [Phi11,   zeros(3),  zeros(3);
                       Phi21,   Phi22,     zeros(3);
                       Phi31,   Phi32,     Phi33    ];
        end
        
        function phi_wedge = wedge_se23(obj, phi)
            %WEDGE_SE23 Wedge operator for SE2(3)
            R = obj.wedge_so3(phi(1:3));
            phi_wedge = [R, phi(4:6), phi(7:9);
                         zeros(2, 5)           ];
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
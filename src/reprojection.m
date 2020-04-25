function A = reprojection(y,y_gnd,allowTranslation)
%% A = reprojection(y,y_gnd,allowTranslation)
% y, y_gnd = 3*n, allowTranslation = 0 or 1. Requires known DA between y
% and y_gnd.
% Gives a best estimatin of transformation matrix A that minimizes the LS
% error between y_gnd and A*y.
% If allowTranslation == 0, A = R*S is a 3 by 3 matrix of scaling and
% rotation.
% If allowTranslation == 1, A is a 4 by 4 matrix of scaling, rotation and
% translation in three directions.
y_gnd_avg = mean(y_gnd,2);
y_avg = mean(y,2);
s1 = norm(y_gnd_avg)/norm(y_avg);
S1 = s1*eye(3);
y_est = S1*y;
[R_1, ~] = icp_3d(y_est, y_gnd);
y_est = R_1*y_est;
x0 = [1,0,0,0];
options = optimoptions('lsqnonlin','Display','iter');
options.Algorithm = 'levenberg-marquardt';
func = @(x) distance3D(x,y_est,y_gnd);
[x,resnorm,residual,exitflag,output] = lsqnonlin(func,x0,[],[],options);
s1 = x(1);
alpha = x(2);
beta = x(3);
gamma = x(4);
Rz = [cos(gamma) -sin(gamma) 0;sin(gamma) cos(gamma) 0;0 0 1];
Ry = [cos(beta) 0 sin(beta);0 1 0;-sin(beta) 0 cos(beta)];
Rx = [1 0 0;0 cos(alpha) -sin(alpha);0 sin(alpha) cos(alpha)];
R = Rz*Ry*Rx;
S = diag([s1 s1 s1]);
A = R*S*R_1*S1;
y_est = A*y;
if allowTranslation
    x0 = [1,0,0,0,0,0,0];
    func = @(x) distanceTrans(x,y_est,y_gnd);
    [x,resnorm,residual,exitflag,output] = lsqnonlin(func,x0,[],[],options);
    s1 = x(1);
    alpha = x(2);
    beta = x(3);
    gamma = x(4);
    xTrans = x(5);
    yTrans = x(6);
    zTrans = x(7);
    T = [xTrans;yTrans;zTrans];
    Rz = [cos(gamma) -sin(gamma) 0;sin(gamma) cos(gamma) 0;0 0 1];
    Ry = [cos(beta) 0 sin(beta);0 1 0;-sin(beta) 0 cos(beta)];
    Rx = [1 0 0;0 cos(alpha) -sin(alpha);0 sin(alpha) cos(alpha)];
    R = Rz*Ry*Rx;
    S = diag([s1 s1 s1]);
    A = [S*R*A T;zeros(1,3) 1];
end
end
function F = distance3D(x,y,y_gnd)
s1 = x(1);
alpha = x(2);
beta = x(3);
gamma = x(4);
Rz = [cos(gamma) -sin(gamma) 0;sin(gamma) cos(gamma) 0;0 0 1];
Ry = [cos(beta) 0 sin(beta);0 1 0;-sin(beta) 0 cos(beta)];
Rx = [1 0 0;0 cos(alpha) -sin(alpha);0 sin(alpha) cos(alpha)];
R = Rz*Ry*Rx;
S = diag([s1 s1 s1]);
y_est = R*S*y;
y_err = y_gnd - y_est;
y_err = y_err.^2;
F = sqrt(y_err(1,:)+y_err(2,:)+y_err(3,:));
end
function F = distanceTrans(x,y,y_gnd)
s1 = x(1);
alpha = x(2);
beta = x(3);
gamma = x(4);
xTrans = x(5);
yTrans = x(6);
zTrans = x(7);
T = [xTrans;yTrans;zTrans];
Rz = [cos(gamma) -sin(gamma) 0;sin(gamma) cos(gamma) 0;0 0 1];
Ry = [cos(beta) 0 sin(beta);0 1 0;-sin(beta) 0 cos(beta)];
Rx = [1 0 0;0 cos(alpha) -sin(alpha);0 sin(alpha) cos(alpha)];
R = Rz*Ry*Rx;
S = diag([s1 s1 s1]);
A = [S*R T;zeros(1,3) 1];
y_aug = [y;ones(1,size(y,2))];
y_est = A*y_aug;
y_est = y_est(1:3,:)./y_est(4,:);
y_err = y_gnd - y_est;
y_err = y_err.^2;
F = sqrt(y_err(1,:)+y_err(2,:)+y_err(3,:));
end
function [R_total, t_total] = icp_3d(p_t, p_s)
k = 0;
R_total = eye(3);
t_total = zeros(3, 1);
for iter = 1:100
    %% Update p2, the current source cloud
    p2 = R_total' * (p_s - t_total);
    
    %% Get p1, the matching points from the target cloud
    match = 1:1:length(p_s);
    for i_s=1:length(p_s)
             dis=p_t(:,i_s)-p_s(:,i_s);
             dis=norm(dis,2);
            min_dist(i_s)=dis;
    end
    p1 = p_t(:, match);
    
    n1=length(p1);
    n2=length(p2);
    
    d=min_dist/std(min_dist);
    Sum_exp=sum(exp(-d));
    w=exp(-d)/Sum_exp;
    %% Centroids of p1 and p2
    mu1 = p1*w';
    mu2 = p2*w';
    
    %% Center the two clouds
    p1_bar = (p1 - mu1).*w;
    p2_bar = (p2 - mu2).*w;
   
    %% Estimate the rotation and translation
    [U, ~, V] = svd(p1_bar * p2_bar');
    R = V*U';
    t = -R*mu1+mu2;
    if det(R)<0
        R=R*[1 0 0;0 1 0;0 0 -1];
    end
    %% Update R_total and t_total
    t_total = R*t_total+t;
    R_total = R*R_total;

end

end
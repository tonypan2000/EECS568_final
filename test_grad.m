clc
%% T2 = odom * T1, where Ti = [si * Ri, 0; ti, 1] is in SIM(3)
p1 = sym('p1', [7, 1], 'real'); % >> T1
p2 = sym('p2', [7, 1], 'real'); % >> T2

%%
s1 = exp(p1(1));
l1 = simplify(norm(p1(2:4)));
X = [0, -p1(4), p1(3)
    p1(4), 0, -p1(2)
    -p1(3), p1(2), 0];
X = X / l1;
R1 = eye(3) + sin(l1) * X + (1 - cos(l1)) * X^2;
R1 = simplify(R1);
t1 = p1(5:7).';

%%
s2 = exp(p2(1));
l2 = simplify(norm(p2(2:4)));
Y = [0, -p2(4), p2(3)
    p2(4), 0, -p2(2)
    -p2(3), p2(2), 0] / l2;
R2 = eye(3) + sin(l2) * Y + (1 - cos(l2)) * Y^2;
R2 = simplify(R2);
t2 = p2(5:7).';

%% 7DOF odom
sigma = p2(1) - p1(1);%simplify(s2 / s1);
R = simplify(R2 * R1.');
t = (t2 - t1) * R1.' / s1;
t = simplify(t);

%% rotation vector
u = [R(3, 2) - R(2, 3)
    R(1, 3) - R(3, 1)
    R(2, 1) - R(1, 2)];
u = simplify(u);
theta = acos((trace(R) - 1) / 2);
theta = simplify(theta);
u = simplify(u / norm(u, 2) * theta);

p = [sigma; u; t.'];
J = jacobian(p, [p1; p2]);

matlabFunction(p, 'Vars', {p1, p2}, 'File', 'calc_odom')
matlabFunction(J, 'Vars', {p1, p2}, 'File', 'calc_measurement_jacob')
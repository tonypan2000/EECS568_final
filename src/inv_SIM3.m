function invT = inv_SIM3(T)
A = T(1:3, 1:3);
t = T(4, 1:3);
s = det(A);
R = A / s;

invT = eye(4);
invT(1:3, 1:3) = R' / s;
invT(4, 1:3) = -t * invT(1:3, 1:3);
end
% This is a simple code to generate a diagonally dominant matrix A
% It implement Neumann series approximations of orders 2, 3 and 4
clc
clear all
close all

% Matrix dimension 64x64
A_dim = 64;
H_dim = 16;

% Generating a tall matrix H such that A = H'*H is diagonally dominant
H = randn(H_dim*A_dim, A_dim);

A = (H'*H / (H_dim*A_dim) * 2^(0));

% Direct (exact) matrix inversion
Ainv = inv(A);

% Isolate diagonal and off-diagonal parts of A
D = diag(diag(A));
E = A - D;
Dinv = inv(D);

% Computing Neumann series approximation of order 2, 3 and 4
Ainv2 = Dinv - Dinv*E*Dinv;
Ainv3 = Dinv - Dinv*E*Dinv + Dinv*E*Dinv*E*Dinv;
Ainv4 = Dinv - Dinv*E*Dinv + Dinv*E*Dinv*E*Dinv - Dinv*E*Dinv*E*Dinv*E*Dinv;

% Approximate inversion errors for comparison
Error2 = norm(Ainv/norm(Ainv) - Ainv2/norm(Ainv2), 'fro');
Error3 = norm(Ainv/norm(Ainv) - Ainv3/norm(Ainv3), 'fro');
Error4 = norm(Ainv/norm(Ainv) - Ainv4/norm(Ainv4), 'fro');

% ======= SAVE MATRICES TO TEXT FILES =======

dlmwrite('A.txt', A, 'delimiter', ' ', 'precision', 16);
dlmwrite('Dinv.txt', Dinv, 'delimiter', ' ', 'precision', 16);
dlmwrite('E.txt', E, 'delimiter', ' ', 'precision', 16);

dlmwrite('Ainv_exact.txt', Ainv,  'delimiter', ' ', 'precision', 16);
dlmwrite('Ainv2.txt',      Ainv2, 'delimiter', ' ', 'precision', 16);
dlmwrite('Ainv3.txt',      Ainv3, 'delimiter', ' ', 'precision', 16);
dlmwrite('Ainv4.txt',      Ainv4, 'delimiter', ' ', 'precision', 16);



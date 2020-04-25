clc
%%
seq = 0;
%data = load(num2str(seq, 'seq%02d_skip2.mat'));
data = load('D:\School\Winter_2020_Shared\ROB_568\Surf_Slam\Surf_Slam_Files\matlab-orb-slam-master\data\seq00_skip2.mat')

vs = data.Map.covisibilityGraph;

numIter = 20;

%%
numPoses = size(vs.Views, 1);
poses = zeros(numPoses, 7);
for k = 1:numPoses
    R = vs.Views.Orientation{k};
    u = [R(3, 2) - R(2, 3)
        R(1, 3) - R(3, 1)
        R(2, 1) - R(1, 2)];
    l_hat = norm(u); 
    if l_hat
        theta = acos((trace(R) - 1) / 2);
        u = theta * u / l_hat;
    else
        u = zeros(3, 1);
    end
    t = vs.Views.Location{k};
    
    poses(k, 1) = log(det(R));
    poses(k, 2:4) = u;
    poses(k, 5:7) = t;
end

%%
numConnections = size(vs.Connections, 1);
odom = zeros(numConnections, 7);
test = zeros(numConnections, 3);
for k = 1:numConnections
    R = vs.Connections.RelativeOrientation{k};
    u = [R(3, 2) - R(2, 3)
        R(1, 3) - R(3, 1)
        R(2, 1) - R(1, 2)];
    l_hat = norm(u); 
    if l_hat
        theta = acos((trace(R) - 1) / 2);
        u = theta * u / l_hat;
    else
        u = zeros(3, 1);
    end
        
    t = vs.Connections.RelativeLocation{k};
    
    if norm(t) == 0
        fprintf('%d, %d\n', vs.Connections.ViewId1(k), vs.Connections.ViewId2(k))
    end
    
    odom(k, 1) = log(det(R));
    odom(k, 2:4) = u;
    odom(k, 5:7) = t;
    
    if k > 1
        
        test(k, 1:3) = test(k-1,1:3) + (R*odom(k, 5:7)')';
    else
         test(k, 1:3) = (R*odom(k, 5:7)')';
    end
end
%plot(test(:,1),test(:,2),'-r',poses(:,5),poses(:,6),'-b')
%%
poses_opt = poses;
A = sparse(7 * (numConnections + 1), 7 * numPoses);
A((7 * numConnections + 1):end, 1:7) = eye(7);
b = zeros(7 * (numConnections + 1), 1);
disp('Optimizing ...')
for i = 1:numIter
    fprintf('[%2d/%2d]\n', i, numIter)
    for k = 1:numConnections
        %idx1 = find(vs.Views.ViewId==vs.Connections.ViewId1(k));
        %idx2 = find(vs.Views.ViewId==vs.Connections.ViewId2(k));
        
        idx1 = vs.Connections.ViewId1(k);
        idx2 = vs.Connections.ViewId2(k);

        if idx1 == 1
            p1 = randn(7, 1) * 1e-8;
        else
            p1 = poses_opt(idx1, :)';
        end
        p2 = poses_opt(idx2, :)';

        J = calc_measurement_jacob(p1, p2);

        A((7 * k - 6):(7 * k), (7 * idx1 - 6):(7 * idx1)) = ...
            J(:, 1:7);
        A((7 * k - 6):(7 * k), (7 * idx2 - 6):(7 * idx2)) = ...
           J(:, 8:14);

        odom_hat = calc_odom(p1, p2);
        
        d = abs(int64(idx1) - int64(idx2)); 
        if d > 1 && d < 10
            l = norm(odom_hat(5:7), 2);
            odom_hat(5:7) = odom_hat(5:7) / l;
        end
% 
%         l = norm(odom_hat(5:7), 2);
%         odom_hat(5:7) = odom_hat(5:7) / l;
              
        % disp([odom_hat'; odom(k, :)])
            
        b((7 * k - 6):(7 * k)) = odom(k, :)' - odom_hat;
    end
    
    delta = A \ b;
    delta = reshape(delta, [7, numPoses])';
    poses_opt = poses_opt + 0.5 * delta;
end

%%
order = colamd(A);
L_reordered = chol(A(:, order)' * A(:, order));
L_original = chol(A' * A);
figure(1)
set(gcf, 'position', [100, 100, 800, 400])
subplot(1, 2, 1)
spy(L_original)
subplot(1, 2, 2)
spy(L_reordered)
print(num2str(seq, 'seq%02d_R.png'), '-r300', '-dpng')

%%
figure(10)
clf()
%for k = 1:numConnections 
%idx1 = find(vs.Views.ViewId==vs.Connections.ViewId1(k));
%idx2 = find(vs.Views.ViewId==vs.Connections.ViewId2(k));
idx1 = vs.Connections.ViewId1;
idx2 = vs.Connections.ViewId2;
plot(...
    [poses(idx1, 5)'; poses(idx2, 5)'], ...
    [poses(idx1, 7)'; poses(idx2, 7)'], 'r')
hold on


plot(poses(:, 5), poses(:, 7), 'k.-')
axis equal
print(num2str(seq, 'seq%02d_before_lc.png'), '-r300', '-dpng')

figure(11)
clf()

plot(...
    [poses_opt(idx1, 5)'; poses_opt(idx2, 5)'], ...
    [poses_opt(idx1, 7)'; poses_opt(idx2, 7)'], 'r')
plot(poses_opt(:, 5), poses_opt(:, 7), 'k.-')
axis equal
print(num2str(seq, 'seq%02d_after_lc.png'), '-r300', '-dpng')

%%
poses_gt = load_gt_poses(seq);
xyz_gt = poses_gt(1:2:end, :, 4)';
xyz_opt = poses_opt(:, 5:7)';
xyz_gt = xyz_gt(:,1:length(xyz_opt));

%xyz_opt = poses_opt(:, 5:7)';

T = reprojection(xyz_opt, xyz_gt, 1);

xyz_opt = T(1:3, 1:3) * xyz_opt + T(1:3, 4);

figure(2)
plot(xyz_gt(1, :), xyz_gt(3, :), 'g', ...
    xyz_opt(1, :), xyz_opt(3, :), 'k', ...
    'linewidth', 2)
axis equal
axis(axis() + [-1, 1, -1, 1] * 10)
legend({'Ground Truth', 'Result'}, 'location', 'best')
title(num2str(seq, 'Sequence %02d'))

delta = xyz_opt - xyz_gt;
err = sqrt(mean(delta(1, :).^2 + delta(2, :).^2 + delta(3, :).^2));
fprintf('Results for sequence %02d:\nRMSE = %.4f\n', ...
    seq, err)
print(num2str(seq, 'seq%02d_result.png'), '-r300', '-dpng')

%%
%{
for k = 1:numPoses
    R = vs.Views.Orientation{k}';
    ex = R(:, 1) * 0.5;
    ez = R(:, 3) * 0.5;
    t = vs.Views.Location{k};
    
    plot(t(1) + [0, ex(1)], t(3) + [0, ex(3)], 'r')
    plot(t(1) + [0, ez(1)], t(3) + [0, ez(3)], 'b')
end
%}
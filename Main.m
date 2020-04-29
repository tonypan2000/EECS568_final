clear;
clc;

%Read in data - Change this to arg for file select?
filename = '2011_10_03_drive_0027_sync';
data = read_kitti(filename);

show_images = true;
if show_images
    images = read_images(filename);
end

% read ground truth
ground_truth = dlmread('poses/00.txt');

%Build mu_init
eul = [data.yaw(1), data.pitch(1), data.roll(1)];
rot = eul2rotm(eul);   % Default eul sequence is zyx
v_body = [data.vf(1);data.vl(1);data.vu(1)];
v_frame = rot*v_body;
%[data.vn(1) data.ve(1)]  Sanity Check
pos = [0; 0; 0];
mu_init = [rot v_frame pos; 0 0 0 1 0; 0 0 0 0 1];


%Build sigma_init and imu_bias_init
sigma_init = 0.01*eye(15);
imu_bias_init = [0;0;0;0;0;0];

%Initilaize Filter
filter = liekf(mu_init, sigma_init, imu_bias_init);

% saves trajectory in se3 as 12 X 1 vectors
poses = [];

% record video
vo = VideoWriter('video', 'MPEG-4');
set(vo, 'FrameRate', 10);
open(vo);

%Main Loop
for t = 2:length(data.lat)
    % Prediction Step
    u = [data.wx(t); data.wy(t); data.wz(t);data.ax(t); data.ay(t); data.az(t)];
    filter.prediction(u);
    
    %Correction Step
    [xEast,yNorth,zUp] = geodetic2enu(data.lat(t),data.lon(t),data.alt(t),data.lat(1),data.lon(1),data.alt(1),wgs84Ellipsoid);
    y = [xEast,yNorth,zUp]';
    filter.correction(y);
    result(t-1,:) = filter.mu_cart;
    
    % appends se3 pose
    poses = [poses; filter.pose];
   
    %Plot
    if show_images
        figure(1); clf;
        subplot(2,1,1)
        hold on; axis equal;
        plot(result(1:t-1,7),result(1:t-1,8))
        plot(result(t-1,7),result(t-1,8),'*')
        xlim([-300 450]);
        ylim([-150 500]);
        hold off

        subplot(2,1,2)
        imshow(images{t})
        drawnow limitrate
        
        F = getframe(gcf);
        writeVideo(vo, F);
    else
        figure(1); clf;
        hold on; axis equal;
        plot(result(1:t-1,7),result(1:t-1,8))
        plot(result(t-1,7),result(t-1,8),'*')
        plot(ground_truth(1:t-1,4),ground_truth(1:t-1,12))
        plot(ground_truth(t-1,4),ground_truth(t-1,12),'-')
        xlim([-300 300]);
        ylim([-50 500]);
        hold off

        drawnow limitrate
    end

end

% plot
% figure(1); clf;
% hold on; axis equal;
% % plot ground truth poses
% plot(ground_truth(1:length(ground_truth),4),ground_truth(1:length(ground_truth),12))  
% % plot our trajectory
% plot(result(1:length(result),7),result(1:length(result),8))
% legend('Ground Truth','Left-InEKF')
% title('Ground Truth vs. Left-Invariant EKF Trajectory')
% xlim([-300 450]);
% ylim([-150 500]);

% saves video
close(vo);

% saves se3 poses trajectory to a .txt file
% writematrix(poses,strcat(filename, '_poses.txt'),'Delimiter',' ')


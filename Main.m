clear;
clc;

%Read in data - Change this to arg for file select?
filename = '0009';
data = read_kitti(filename);

show_images = true;
if show_images
    images = read_images(filename);
end

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
        figure(1); clf; hold on;
        subplot(2,1,1)
        hold on 
        plot(result(1:t-1,7),result(1:t-1,8))
        plot(result(t-1,7),result(t-1,8),'*')
        xlim([-100 250])
        ylim([-100 100])
        hold off

        subplot(2,1,2)
        imshow(images{t})
        drawnow limitrate
    else
        figure(1); clf; hold on;
        plot(result(1:t-1,7),result(1:t-1,8))
        plot(result(t-1,7),result(t-1,8),'*')
        xlim([-100 250])
        ylim([-100 100])
        drawnow limitrate
    end

end

% saves se3 poses trajectory to a .txt file
writematrix(poses,strcat(filename, '_poses.txt'),'Delimiter',' ')


clear all;
close all;
clc;

%% User setup

kf = true; %Set to false to use visual odometry

isPlot = true;

sequence = 0;

imageDir = ['dataset' filesep 'sequences' filesep num2str(sequence,'%02d') filesep 'image_0'];
imageExt = '.png';

calibFile = ['dataset' filesep 'sequences' filesep num2str(sequence,'%02d') filesep 'calib.txt'];
cameraID = 0;

codewords = load(['dataset' filesep 'codewords.mat']);
codewords = codewords.codewords;



%% Get feature vocabulary

% if no vocabulary exists
	% Load set of images
	% Create vocabulary
	% Save vocabulary
% else
	% Load vocabulary
% end


%% Setup Global variables

global Map;
Map.covisibilityGraph = ourViewSet();

global State;
State.mu = [0;0;0];
State.Sigma = zeros(length(State.mu));

global Params;
Params.theta = 15; % Number of shared observations a keyframe must have to be considered the same map points
Params.theta_min = 100; % Defines high covisability for spanning tree
Params.keyFramePercentOfBestScoreThreshold = 75; % bag of words returns keyframes that are more than this percentage of the best match
Params.cameraParams = load_camera_params(calibFile, cameraID);
Params.minMatchesForConnection = 50;
% ADD number of features to say we didn't lose localization
% ADD angle threshold between v and n
% ADD scale invariance region - perhaps set from data set

Params.cullingSkip = 25;
Params.cullingThreshold = 0.9;

Params.kdtree = KDTreeSearcher(codewords);
Params.numCodewords = size(codewords, 1);
Params.numFramesApart = 20;

Params.numViewsToLookBack = 5;
Params.minMatchRatioRatio = 0.4;

Params.numSkip = 2;
Params.deletedframes = [];

global Debug;
Debug.displayFeaturesOnImages = false;

%% Initialize Left Invarient Extended Kalman Filter
sensor_data = read_kitti('dataset/sensor_data',Params.numSkip);

%Build mu_init
eul = [sensor_data.yaw(1), sensor_data.pitch(1), sensor_data.roll(1)];
rot = eul2rotm(eul);   % Default eul sequence is zyx
v_body = [sensor_data.vf(1);sensor_data.vl(1);sensor_data.vu(1)];
v_frame = rot*v_body;
pos = [0; 0; 0];
mu_init = [rot v_frame pos; 0 0 0 1 0; 0 0 0 0 1];


%Build sigma_init and imu_bias_init
sigma_init = 0.01*eye(15);
imu_bias_init = [0;0;0;0;0;0];

%Initilaize Filter
liekf_filter = liekf(mu_init, sigma_init, imu_bias_init);


%% Run ORB-SLAM

imagesFiles = dir([imageDir, filesep, '*', imageExt]);
framesToConsider = 1:Params.numSkip:length(imagesFiles);
frames = cell([1 length(framesToConsider)]);
 
for i = 1:length(framesToConsider)
	frameIdx = framesToConsider(i);
	frames{i} = imread([imagesFiles(frameIdx).folder, filesep, imagesFiles(frameIdx).name]);
end

liekf_results = zeros(length(framesToConsider),9);

for i = 1:length(framesToConsider)

	if iscell(frames)
		frame = frames{i};
	else
		frame = frames(i);
    end
    
    if i > 1
        % Prediction Step
        liekf_u = [sensor_data.wx(i); sensor_data.wy(i); sensor_data.wz(i);sensor_data.ax(i); sensor_data.ay(i); sensor_data.az(i)];
        liekf_filter.prediction(liekf_u);

        %Correction Step
        [xEast,yNorth,zUp] = geodetic2enu(sensor_data.lat(i),sensor_data.lon(i),sensor_data.alt(i),sensor_data.lat(1),sensor_data.lon(1),sensor_data.alt(1),wgs84Ellipsoid);
        ekf_y = [xEast,yNorth,zUp]';
        liekf_filter.correction(ekf_y);
        liekf_results(i,:) = liekf_filter.mu_cart;
    end
	surf_slam(frame,i,liekf_results(i,:),kf);

    fprintf('Sequence %02d [%4d/%4d]\n', ...
        sequence, i, length(framesToConsider))
end

save([num2str(sequence, 'data/seq%02d'), ...
    num2str(Params.numSkip, '_skip%d.mat')], 'Map')

%% Display

if isPlot
    camPoses = poses(Map.covisibilityGraph);
	figure
	hold on
	traj = cell2mat(camPoses.Location);
    x = traj(:, 1);
    z = traj(:, 3);
    plot(x, z, 'x-')
    axis equal
	grid on
    
    %{
	validIdx = sqrt(xyzPoints(:, 1).^2 + xyzPoints(:, 2).^2 + xyzPoints(:, 3).^2) < 100;
	validIdx = validIdx & (xyzPoints(:, 3) > 0);

	pcshow(xyzPoints(validIdx, :), 'VerticalAxis', 'y', 'VerticalAxisDir', 'down', ...
		'MarkerSize', 45);
    %}

    %validIdx = sqrt(xyzPoints(:, 1).^2 + xyzPoints(:, 2).^2 + xyzPoints(:, 3).^2) < 500;
    %scatter(xyzPoints(validIdx, 1), xyzPoints(validIdx, 3), '.')
	hold off;
end

figure
plot(liekf_results(:,7),liekf_results(:,8))

optimize_graph;

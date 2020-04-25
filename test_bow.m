function test_bow(seq)

if ~exist('seq', 'var')
    seq = 4;
end

data = load('dataset/codewords.mat');

kdtree_mdl = KDTreeSearcher(data.codewords);

numCodewords = size(data.codewords, 1);

%%
skip = 3;
if exist(num2str(seq, 'dataset/sequences/%02d/image_0.mat'), 'file')
    disp('Using precalculated SURF features ...')
    data = load(num2str(seq, 'dataset/sequences/%02d/image_0.mat'));
    features = data.features(1:skip:end);
    points = data.validPoints(1:skip:end);
    
    numImages = length(features);
    
    bow = zeros(numImages, numCodewords);
    parfor k = 1:numImages       
        bow(k, :) = calc_bow_repr(features{k}, kdtree_mdl, numCodewords);
    end
else
    imageFiles = dir(num2str(seq, 'dataset/sequences/%02d/image_0/*.png'));
    imageFiles = imageFiles(1:skip:end);
    numImages = length(imageFiles);
    
    bow = zeros(numImages, numCodewords);
    features = cell(numImages, 1);
    points = cell(numImages, 1);
    parfor k = 1:numImages
        frame = imread([imageFiles(k).folder, '/', imageFiles(k).name]);
        
        points{k} = detectSURFFeatures(frame);
        [features{k}, points{k}] = extractFeatures(frame, points{k});
        
        bow(k, :) = calc_bow_repr(features{k}, kdtree_mdl, numCodewords);
    end
end

%%
loop_closure_proposal = zeros(numImages, 1);
num_frames_apart = 50;
matchRatio = zeros(numImages, 1);
for i = (num_frames_apart + 1):numImages
    d2 = hist_diff(bow(1:(i - num_frames_apart), :), bow(i, :));
    
    [~, j] = min(d2);
    
    matchedIdx = matchFeatures(features{i}, features{j}, 'unique', true);
    
    ni = length(features{i});
    nj = length(features{j});
    matchRatio(i) = numel(matchedIdx) / (ni + nj);
    
    if matchRatio(i) > 0.2
        loop_closure_proposal(i) = j;
    end
end

%%
poses_gt = load_gt_poses(seq);
poses_gt = poses_gt(1:skip:end, :, :);

x_gt = poses_gt(1:numImages, 1, 4);
z_gt = poses_gt(1:numImages, 3, 4);

%%
%{
vs = viewSet();

for i = 1:numImages
    vs = addView(vs, i, 'Points', points{i}, ...
        'Orientation', squeeze(poses_gt(i, :, 1:3))', ...
        'Location', reshape(poses_gt(i, :, 4), [1, 3]));
    
    if i > 1
        vs = addConnection(vs, i, i - 1);
    end
    
    j = loop_closure_proposal(i);
    if j > 0
        vs = addConnection(vs, i, j);
    end
end

tracks = findTracks(vs);
%}
%%
figure(1)
clf()
plot(x_gt, z_gt, 'k')
hold on
axis equal
axis(axis() + [-10, 10, -10, 10])

for i = 1:numImages
    j = loop_closure_proposal(i);
    if j > 0
        plot([x_gt(i), x_gt(j)], [z_gt(i), z_gt(j)], 'r')
    end
end
end

%%
function repr = calc_bow_repr(features, kdtree_mdl, num_codewords)
idx = knnsearch(kdtree_mdl, features);

repr = histcounts(idx, 1:(num_codewords + 1));
repr = repr / sum(repr);
end

%%
function d2 = hist_diff(h1, h2)
d2 = sum((h1 - h2).^2 ./ (h1 + h2 + 1e-6), 2);
end

%%
function poses = load_gt_poses(seq)
%   Load ground truth poses as a sequence of homogeneous transforms.
%
%   Input:
%   -   sequence
%       int
%       A sequence ID in 'dataset/poses'.
%
%   Output:
%   -   poses
%       [N, 3, 4] double
%       poses(i, :, :) = [R(i), t(i)], where R(i) is a 3x3 rotation matrix
%       and t(i) is a 3x1 translation vector.

poses = dlmread(num2str(seq, 'dataset/poses/%02d.txt'));
poses = reshape(poses, [size(poses, 1), 4, 3]);
poses = permute(poses, [1, 3, 2]);
end

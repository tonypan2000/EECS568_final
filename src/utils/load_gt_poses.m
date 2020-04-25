function poses = load_gt_poses(sequence)
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

file_name = ['dataset/poses/', num2str(sequence, '%02d'), '.txt'];
file_id = fopen(file_name, 'r');

formatSpec = '%f';
poses = fscanf(file_id, formatSpec);

fclose(file_id);

n = numel(poses) / 12;
poses = reshape(poses, [4, 3, n]);
poses = permute(poses, [3, 2, 1]);
end
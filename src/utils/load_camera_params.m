function cameraParams = load_camera_params(path, camera_id)

P = dlmread(path, ' ', 0, 1);
P = reshape(P(camera_id + 1, :), [4, 3]);

K = P(1:3, 1:3);
t = P(4, :) / K;

cameraParams = cameraParameters('WorldUnits', 'm',...
    'IntrinsicMatrix', K, ...
    'RotationVectors', zeros(1, 3), ...
    'TranslationVectors', t);

end
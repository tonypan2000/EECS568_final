%   Precalculate SURF features and save them to `dataset/sequences/**/image_0.mat`

for s = 0:21
    % Use only images from the left camera.
    files = dir(['dataset/sequences/', num2str(s, '%02d'), '/image_0/*.png']);
    n_images = length(files);
    
    features = cell(n_images, 1);
    validPoints = cell(n_images, 1);
    
    fprintf(['Calculating SURF features for sequence `', num2str(s, '%02d'), '` ...\n'])
    parfor k = 1:n_images
        path = [files(k).folder, '/', files(k).name];
        img = imread(path);
        
        points = detectSURFFeatures(img);
        [features{k}, validPoints{k}] = extractFeatures(img, points);
    end
    
    path = ['dataset/sequences/', num2str(s, '%02d'), '/image_0.mat'];
    
    fprintf('Saving SURF features to `%s` ...\n', path)
    save(path, 'features', 'validPoints')
    fprintf('Done.\n')
end
function codewords = gen_codewords(skip, num_codewords)

if ~exist('skip', 'var')
    skip = 100;
end

if ~exist('codewords', 'var')
    num_codewords = 64;
end

vocab = cell(9, 1);

parfor seq = 0:8
    file = num2str(seq, 'dataset/sequences/%02d/image_0.mat');
    data = load(file);
    
    features = cell2mat(data.features);
    num_frames = numel(data.features);
    
    fprintf('Sequence %02d: %d frames, %4.2f points per frame\n', ...
        seq, num_frames, size(features, 1) / num_frames)
    
    vocab{seq + 1} = features(1:skip:end, :);
end

vocab = cell2mat(vocab);

fprintf('Total number of vocabularies = %d\n', size(vocab, 1))

disp('Clustering ... (this might take some time)')
[~, codewords] = kmeans(vocab, num_codewords);

disp('Saving ...')
save('dataset/codewords.mat', 'codewords')

end
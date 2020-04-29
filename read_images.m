function input_photos = read_images(dataset)

    directory = strcat(dataset, '/image_03/data/');
    input_files = dir(directory);
    cnt = 1;
    for i=3:length(input_files)
        input_photos{cnt} = imread(strcat(directory,input_files(i).name));
        cnt = cnt + 1;
    end
end
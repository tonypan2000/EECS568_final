function input_photos = read_images(dataset)

    directory = strcat(dataset, '/photos/');
    input_files = dir(directory);
    cnt = 1;
    for i=1:length(input_files)
        input_files(i);
        if length(input_files(i).name) > 5
            input_photos{cnt} = imread(strcat(directory,input_files(i).name));
            cnt = cnt + 1;
        end
    end
end
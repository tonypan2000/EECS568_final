function input_data = read_kitti(dataset)
    directory = strcat(dataset, '/data/');
    input_files = dir(directory);
    input_data = raw_data();
    for i=3:length(input_files)
        filename = input_files(i).name;
        current_file = fopen(strcat(directory, filename));
        raw_input = textscan(current_file, '%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %d %d %d %d %d');
        fclose(current_file);
        input_data.lat = [input_data.lat cell2mat(raw_input(1))];
        input_data.lon = [input_data.lon cell2mat(raw_input(2))];
        input_data.alt = [input_data.alt cell2mat(raw_input(3))];
        input_data.roll = [input_data.roll cell2mat(raw_input(4))];
        input_data.pitch = [input_data.pitch cell2mat(raw_input(5))];
        input_data.yaw = [input_data.yaw cell2mat(raw_input(6))];
        input_data.vn = [input_data.vn cell2mat(raw_input(7))];
        input_data.ve = [input_data.ve cell2mat(raw_input(8))];
        input_data.vf = [input_data.vf cell2mat(raw_input(9))];
        input_data.vl = [input_data.vl cell2mat(raw_input(10))];
        input_data.vu = [input_data.vu cell2mat(raw_input(11))];
        input_data.ax = [input_data.ax cell2mat(raw_input(12))];
        input_data.ay = [input_data.ay cell2mat(raw_input(13))];
        input_data.az = [input_data.az cell2mat(raw_input(14))];
        input_data.af = [input_data.af cell2mat(raw_input(15))];
        input_data.al = [input_data.al cell2mat(raw_input(16))];
        input_data.au = [input_data.au cell2mat(raw_input(17))];
        input_data.wx = [input_data.wx cell2mat(raw_input(18))];
        input_data.wy = [input_data.wy cell2mat(raw_input(19))];
        input_data.wz = [input_data.wz cell2mat(raw_input(20))];
        input_data.wf = [input_data.wf cell2mat(raw_input(21))];
        input_data.wl = [input_data.wl cell2mat(raw_input(22))];
        input_data.wu = [input_data.wu cell2mat(raw_input(23))];
        input_data.pos_accuracy = [input_data.pos_accuracy cell2mat(raw_input(24))];
        input_data.vel_accuracy = [input_data.vel_accuracy cell2mat(raw_input(25))];
        input_data.navstat = [input_data.navstat cell2mat(raw_input(26))];
        input_data.numsats = [input_data.numsats cell2mat(raw_input(27))];
        input_data.posmode = [input_data.posmode cell2mat(raw_input(28))];
        input_data.velmode = [input_data.velmode cell2mat(raw_input(29))];
        input_data.orimode = [input_data.orimode cell2mat(raw_input(30))];
    end
    
    format_spec = 'yyyy-MM-dd HH:mm:ss.SSSSSS';
    file = fopen(strcat(dataset, '/timestamps.txt'));
    line = fgetl(file);
    while ischar(line)
        raw_input = datetime(line, 'InputFormat', format_spec);
        input_data.timestamp = [input_data.timestamp raw_input];
        line = fgetl(file);
    end
    fclose(file);
end
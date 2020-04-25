clc
close all
%%
for k = 0:8
    figure(k + 1)
    poses = load_gt_poses(k);
    x = squeeze(poses(:, 1, 4));
    z = squeeze(poses(:, 3, 4));
    plot(x, z, 'k', 'linewidth', 1)
    axis equal
    print(num2str(k, 'seq%02d.png'), '-dpng', '-r300')
end
% use data "display_neurons.mat"
% created at 2019-7-10

%% make gif
gifsize=[5 5];
clf;shine;axis off;
gifmaker('init','demo.gif',gifsize,0.5);
for te = 0:10:355
    clf;shine;axis off;
    tree_temp = rot_tree(demo_trees(1), [0 te 0]);
    plot_tree_color(tree_temp, 'green');
    gifmaker('loop','demo.gif',gifsize,0.5);
end
gifmaker('finish','demo.gif',gifsize,0.5);

%% convert gif to movie
gifImag=imread('demo.gif','Frames','all');
v = VideoWriter('demo.avi','Uncompressed AVI');
v.FrameRate=3;
open(v);writeVideo(v,gifImag);close(v);
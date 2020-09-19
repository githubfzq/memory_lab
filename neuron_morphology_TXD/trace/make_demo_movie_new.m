% use data "display_neurons.mat"
% created at 2019-7-10
% not by creating gif

first_frame_size=[0 0;0 0];
v = VideoWriter('demo_new.mp4','MPEG-4');
v.FrameRate=25;
open(v);
for te = 0:2:360
    cla;axis off;
    tree_temp = rot_tree(demo_trees(1), [0 te 0]);
    plot_tree_color(tree_temp,gfpcolor);
    shine;
    curXLim=get(gca,'XLim');curYLim=get(gca,'YLim');
    % make sure all frames are the same size
    if first_frame_size(1,1)==0 && first_frame_size(2,1)==0
        first_frame_size=[curXLim;curYLim];
    else
        xlim(first_frame_size(1,:));ylim(first_frame_size(2,:));
    end 
    curFrame=getframe(gca);
    writeVideo(v,curFrame);
end
close(v);
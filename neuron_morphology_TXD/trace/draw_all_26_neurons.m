%% data named as "display_neurons.mat"

colors_26=[repmat({gfpcolor},[1,16]),repmat({gfpNearColor},[1,10])];
plot_multi_tree([RotedTree.Gfp,RotedTree.GfpNear],colors_26,[4,8]);
draw_scale_bar(100,'xy');
figure;
plot_multi_tree([RotedTree.Gfp,RotedTree.GfpNear],colors_26,[4,8],mat2cell(1:26,1,ones([1,26])));
draw_scale_bar(100,'xy');

%% plot 2 demo neurons
demo_trees=[RotedTree.GfpNear(8),RotedTree.Gfp(3)]; % gfp+:[3],gfp-:[8] change order on July,17,2019
color_demo=[{gfpNearColor},{gfpcolor}]; % change order on July,17,2019
figure;
plot_multi_tree(demo_trees,color_demo,[1,2]);
draw_scale_bar(100,'xy');
% adjust window width and height
f=gcf;
f.Position([3,4])=600*[40,30]/25.4; % mm to inch to pixel
f.Position([1,2])=[100,100]; % move figure bottom-left
% move scalebar
f.Children(1).Position([1,2])=[0.78,0.05];
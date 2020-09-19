gfp_sholl_profile=cellfun(@(x) sholl_tree(x,20),gfp_trees,'UniformOutput',false);
near_sholl_profile=cellfun(@(x) sholl_tree(x,20),near_trees,'UniformOutput',false);
intersect_max_n=max(cellfun(@(x) max(size(x)),[gfp_sholl_profile near_sholl_profile]));
sholl_data=zeros([15,intersect_max_n])'; %前9列为GFP+，后6列为GFP+ near
for n=1:9
    sholl_data(1:max(size(gfp_sholl_profile{n})),n)=(gfp_sholl_profile{n})';
end
[~,d]=sholl_tree(gfp_trees{1},20);r=d'/2;
mean_inter_gfp=mean(sholl_data(:,1:9),2);
mean_inter_near=mean(sholl_data(:,10:end),2);
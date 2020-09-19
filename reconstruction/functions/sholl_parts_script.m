% absolute Sholl radius
sholl_res.gfp=arrayfun(@sholl_parts,DefinedTrees.Gfp.whole,...
    ones(size(DefinedTrees.Gfp.whole)),'UniformOutput',false);
sholl_res.gfpNear=arrayfun(@sholl_parts,DefinedTrees.GfpNear.whole,...
    ones(size(DefinedTrees.GfpNear.whole)),'UniformOutput',false);
sholl_tab.gfp=[];sholl_tab.gfpNear=[];
for n=1:length(sholl_res.gfp)
    sholl_tab.gfp=[sholl_tab.gfp;sholl_res.gfp{n}];
end
for n=1:length(sholl_res.gfpNear)
    sholl_tab.gfpNear=[sholl_tab.gfpNear;sholl_res.gfpNear{n}];
end
writetable(sholl_tab.gfp,'Sholl of GFP+.csv');
writetable(sholl_tab.gfpNear,'Sholl of GFP-.csv');

% normalized Sholl radius
sholl_norm.gfp=arrayfun(@sholl_normalized,DefinedTrees.Gfp.whole,...
    ones(size(DefinedTrees.Gfp.whole)),'UniformOutput',false);
sholl_norm.gfpNear=arrayfun(@sholl_normalized,DefinedTrees.GfpNear.whole,...
    ones(size(DefinedTrees.GfpNear.whole)),'UniformOutput',false);
sholl_norm_tab.gfp=[];sholl_norm_tab.gfpNear=[];
for n=1:length(sholl_norm.gfp)
    sholl_norm_tab.gfp=[sholl_norm_tab.gfp;sholl_norm.gfp{n}];
end
for n=1:length(sholl_norm.gfpNear)
    sholl_norm_tab.gfpNear=[sholl_norm_tab.gfpNear;sholl_norm.gfpNear{n}];
end
writetable(sholl_norm_tab.gfp,'Normalized Sholl of GFP+.csv');
writetable(sholl_norm_tab.gfpNear,'Normalized Sholl of GFP-.csv');
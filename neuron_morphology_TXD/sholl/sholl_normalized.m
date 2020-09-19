% function table=sholl_normalized(tree,percent)

function tb=sholl_normalized(tree,percent)
    max_r=max(eucl_tree(tree));
    r_step=max_r*percent*0.01;
    tb=sholl_parts(tree,r_step);
    tb=tb(tb.intersections>0,:);
    tb.radius_normalized=tb.radius/max_r;
end
    
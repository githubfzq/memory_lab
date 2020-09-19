%% format swc file to adjust neurom package in python.
function format_swc(swcFile)
    tree=load_tree(swcFile);
    if ~is_soma_root(tree)
        tree=redirect_to_soma(tree);
    end
    if ~isSubtreeUniform(tree)
        tree=find_new_soma(tree);
    end
    tree=sort_tree(tree);
    exportSwc(tree,swcFile);
end
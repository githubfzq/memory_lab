% Export neuron in TreesToolbox to swc file, after converting the region ('tree.R') to rname.
% Use this function to repair the problem that "swc_tree" function in TreesToolbox do not regard tree.R to tree.R(tree.rnames). rname=='1' =>
% written by FanZuquan on Dec,8,2019

function exportSwc(tree, filename)
    tree.R=arrayfun(@(x) str2num(tree.rnames{x}), tree.R);
    sorted=sort_tree(tree);
    swc_tree(sorted, filename);
end
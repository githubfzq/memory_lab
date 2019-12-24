% redirect a neuron tree to soma point
% format: outTree=redirect_to_soma(inTree)
function outTree=redirect_to_soma(inTree)
    somaRInd=find(ismember(inTree.rnames,'1')); % tree.rnames(somaRInd)='1'
    somaInd=find(inTree.R==somaRInd);
    outTree=redirect_tree(inTree,somaInd);
end
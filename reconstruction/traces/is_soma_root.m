% Detect whether the soma is the root of a neuron tree.
% Format: result = is_soma_root(tree)
% result: locical 1(true) or 0(false)
% by FanZuquan Dec,4,2019
function res=is_soma_root(tree)
    somaRInd=find(ismember(tree.rnames,'1')); % tree.rnames(somaRInd)='1'
    somaInd=find(tree.R==somaRInd);
    ipar=ipar_tree(tree);
    res=min(ipar(somaInd,2))==0; % ipar(somaInd,:)=somaInd,0,0,0,...
end
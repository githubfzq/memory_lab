% Make the radii of multiple soma points the same,  which equal to the mean radii.
% format: outTree = makeSomaRadiiSame(inTree)

function outtr=makeSomaRadiiSame(intr)
    somaRInd=find(ismember(intr.rnames,'1')); 
    somaInd=find(intr.R==somaRInd);
    outtr=intr;
    if length(somaInd)>1
        somaD=ones(size(somaInd))*mean(intr.D(somaInd));
        outtr.D(somaInd)=somaD;
    end
end
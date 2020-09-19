vFilaments = aImarisApplication.GetFactory().CreateFilaments();
vPositions = [[1, 2, 0],  [2, 4, 0],  [4, 4, 0],  [6, 3, 0],  [5, 5, 0],  [9, 6, 0],  [5, 8, 0],  [6, 9, 0],  [13, 5, 0],  [14, 7, 0],  [17, 9, 0]];
vNumberOfPointsPerFilament = [8, 3];
vRadii     = [0.5, 0.5, 0.5, 0.8, 0.5, 0.2, 0.3, 0.8, 0.3, 0.7, 0.9];
vTypes     = [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0];
vEdges     = [[0, 1], [1, 2], [2, 3], [2, 4], [4, 5], [4, 6], [6, 7], [0,1], [1,2]];
vNumberOfEdgesPerFilament = [7, 2];
vTimeIndices = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
vTimeIndexPerFilament = [0, 0];
vFilaments.AddFilamentsList(vPositions,vNumberOfPointsPerFilament,...
    vRadii,vTypes,vEdges,vNumberOfEdgesPerFilament,vTimeIndexPerFilament);
vScene = aImarisApplication.GetSurpassScene();
vScene.AddChild(vFilaments, -1);
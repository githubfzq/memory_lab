%% 2019-7-15
imaris_filament_to_swc('20190620_neuron3.swc');
repair_imaris_tif_voxelsize('xy',0.31);
repair_imaris_tif_voxelsize('z',1.5);
tr(1)=load_tree('20190620_neuron3.swc');
tr(1)=scale_tree(tr(1),[0.31 0.31 1.5]);
swc_import_to_imaris(tr(1));
imaris_filament_to_swc('20190620_neuron3.swc'); % fix size of image and tree
tr(1)=load_tree('20190620_neuron3.swc');
treeColor=[0 0.4275 0.1725];treeColor2=[0.1451 0.1451 0.1451];
plot_tree_color(quaddiameter_tree(apical_upside(tr(1))),treeColor);
print('20190620_neuron3','-dpng');
 
imaris_filament_to_swc('20190621_neuron5.swc');
repair_imaris_tif_voxelsize('xy',0.31);
repair_imaris_tif_voxelsize('z',1.5);
tr(2)=load_tree('20190621_neuron5.swc');
tr(2)=scale_tree(tr(2),[0.31,0.31,1.5]);
swc_import_to_imaris(tr(2));
imaris_filament_to_swc('20190621_neuron5.swc');
tr(2)=load_tree('20190621_neuron5.swc');
plot_tree_color(quaddiameter_tree(apical_upside(tr(2))),treeColor);
print('20190621_neuron5','-dpng');

imaris_filament_to_swc('20190630_neuron2.swc');
repair_imaris_tif_voxelsize('xy',0.198);
repair_imaris_tif_voxelsize('z',1.5);
tr(3)=load_tree('20190630_neuron2.swc');
tr(3)=scale_tree(tr(3),[0.198,0.198,1.5]);
swc_import_to_imaris(tr(3));
imaris_filament_to_swc('20190630_neuron2.swc');
tr(3)=load_tree('20190630_neuron2.swc');
plot_tree_color(quaddiameter_tree(apical_upside(tr(3))),treeColor);
print('20190630_neuron2','-dpng');

save morpho_data.mat

%% 2019-9-2

load morpho_data.mat

% Export trees reconstructed today
imaris_filament_to_swc('20190630_neuron1.swc');
imaris_filament_to_swc('20190701_neuron4.swc');
imaris_filament_to_swc('20190723_neuron1.swc');
imaris_filament_to_swc('20190723_neuron3.swc');
imaris_filament_to_swc('20190726_neuron4.swc');
imaris_filament_to_swc('20190823_neuron1.swc');
imaris_filament_to_swc('20190823_neuron2.swc');

% Load into tr
swcFiles=ls('*.swc');
swcToday=swcFiles([3,5:end],:);
for n=4:10
    tr(n)=load_tree(swcToday(n-3,:));
end

save morpho_data.mat

%% 2019-9-4 & 2019-10-19

load morpho_data.mat

% Re-export swc files with new export function
imaris_filament_to_swc('20190620_neuron3.swc');
imaris_filament_to_swc('20190621_neuron5.swc');
imaris_filament_to_swc('20190630_neuron1.swc');
imaris_filament_to_swc('20190630_neuron2.swc');
imaris_filament_to_swc('20190701_neuron4.swc');
imaris_filament_to_swc('20190723_neuron1.swc');
imaris_filament_to_swc('20190723_neuron3.swc');
imaris_filament_to_swc('20190726_neuron4.swc');
imaris_filament_to_swc('20190823_neuron1.swc');
imaris_filament_to_swc('20190823_neuron2.swc');

for n=1:size(swcFiles,1)
    tr(n)=load_tree(swcFiles(n,:));
end

% plot to validate dendrites region 
for n=1:10
    figure;
    plot_tree(tr(n),tr(n).R);
end

save morpho_data.mat
%% 2019-12-4

% repair soma point is not root (for python import)
load morpho_data.mat
for n=1:10
    if ~is_soma_root(tr(n))
        tr(n)=redirect_to_soma(tr(n));
    end
    if ~isSubtreeUniform(tr(n))
        tr(n)=find_new_soma(tr(n));
    end
    tr(n)=sort_tree(tr(n));
    exportSwc(tr(n),[tr(n).name,'.swc']);
end
save morpho_data.mat
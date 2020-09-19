%% 2020-5-17 data copied from TXD
addpath('..\..\traces\')

gfpFiles=dir('GFP+\*.swc');
controlFiles=dir('GFP-\*.swc');

cd GFP+
for n=1:size(gfpFiles,1)
    gfpTr(n)=load_tree(gfpFiles(n).name);
end
cd ..\GFP-
for n=1:size(controlFiles,1)
    controlTr(n)=load_tree(controlFiles(n).name);
end
cd ..

%% reload new repaired data
gfpFiles=dir('GFP+_repair\*.swc');
controlFiles=dir('GFP-_repair\*.swc');

cd GFP+_repair
for n=1:size(gfpFiles,1)
    gfpTr(n)=load_tree(gfpFiles(n).name);
end
cd ..\GFP-_repair
for n=1:size(controlFiles,1)
    controlTr(n)=load_tree(controlFiles(n).name);
end
cd ..
save tree_data.mat
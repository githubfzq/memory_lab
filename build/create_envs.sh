for envfile in "$1/environment.*.yml"; do
    basename=$(basename $envfile)
    envname="${basename##environment.}"
    envname="${envname%%.yml}"
    echo creating conda environment $envname from $basename...
    conda env create -f $envfile 
    conda run -n $envname python -m ipykernel install --prefix /opt/conda --name $envname --display-name "Python ($envname)" 
done

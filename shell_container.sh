export APPTAINER_CMD=/disk/users/`whoami`/apptainer/bin/apptainer
export PROJECTS_DIR="$(dirname "$PWD")"
export SAVEDIR=/scratch/`whoami`
apptainer exec --nv -B /scratch/`whoami` -B /disk/users/`whoami` -B /home/hep/`whoami` \
            /disk/users/lprate/containers/MuonsAndMatterContainer.sif 

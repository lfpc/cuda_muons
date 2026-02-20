USER_SITE=$(python3 -m site --user-site)
export PYTHONPATH="$USER_SITE:$PYTHONPATH"
export PYTHONPATH=$PYTHONPATH:`readlink -f geant4/cpp/build`
# CUDA Muon Sampling Simulation

Lightweight tools to collect single-step Geant4 samples, build CUDA-friendly histograms, and run a GPU-accelerated muon sampler.

## Quick start

Prerequisites
- Python 3.8+ and pip
- CUDA (if you plan to run the CUDA simulation)
- Project environment: ensure PROJECTS_DIR is set or run from the project root so relative paths resolve.

If you are using uzh-physik cluster, you can use the same container used for muons_and_matter, i.e., shell the container with 

```
cd ..
shell_container.sh
```

To be able to run the CUDA code, one needs first to install the source code. You can do that by executing the script `install_cuda.sh`. The installation currently happens locally, (by using pip install --user). If you are using the container, be sure to execute this step (inside the container) when running for the first time. You won't need to do it again, unless you wish to modify the cuda soruce code.

## Sampling data

In [data](data), one can find the histograms for some materials. If one wish to sample from different materials, one simply needs to do the following procedure, specifying the material as its Geant4 name
(the following is an example to generate the histograms for iron):

1. Collect single-step sampling data from Geant4 outputs:
   - Run the extractor script:
     ```
     python3 utils_cuda_muons/collect_single_step_data.py --material G4_Fe
     ```

2. Build histograms used by the CUDA sampler:
   - Run:
     ```
     python3 utils_cuda_muons/build_histograms.py --alias --material G4_Fe
     ```


## Running a simulation

   One can launch the main simulator by simply running
     ```
     python3 cuda_muons.py
     ```

  Be aware of the possible arguments (run `python3 cuda_muons.py -h`). The construction of the geometry is the same as in muons_and_matter (refer to the [main README file](../README.md)).


# RTDBStream

## Prerequisites
Your GPU must support NVIDIA RTX (hardware raytracing acceleration). For consumer GPUs, this applies to NVIDIA RTX 2000, 3000, and 4000 series, but there are workstation and data-center GPUs that also support RTX.

## Configuration

- Edit script/run.py to set experiment parameters (dataset, window, stride, Eps, MinPts, etc.)

- Run all experiments
    ```
    $ cd .
    $ mkdir build
    $ python script/run.py 
    ```
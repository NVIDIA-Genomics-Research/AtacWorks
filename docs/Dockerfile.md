# AtacWorks Dockerfile

All the AtacWorks SDK components can be executed inside a docker environment. For ease of use,
the AtacWorks team has created a public docker image that can be pulled to run the toolkit.

## Install Docker with GPU support

Docker now ships with native support for GPU. Please follow the instructions on the [nvidia-docker](https://github.com/nvidia/nvidia-docker/wiki/Installation-(Native-GPU-Support))
page to setup the docker framework correctly.

## Test AtacWorks docker setup
If the setup above completed successfully and your system has access to [Docker Hub](https://hub.docker.com/r/claraomics/atacworks),
then the following command will run a sample AtacWorks workflow.

```
    docker run --gpus all claraomics/atacworks /AtacWorks/tests/end-to-end/run.sh
```

If the above command doesn't run successfully, please stop and re-install docker or contact the
AtacWorks dev team on GitHub for more help.

## Run custom workflow in docker
Before starting on custom workflows please refer to the tutorials available in the README of the repository. Those will
help you get familiarized with the features of the SDK.
Once you are comfortable with them, you can use the docker environment to run all of the commands with ease.

* Mount volumes to the container
Use the `-v` option in docker to mount volumes with your data. Official documentation for the options can be found [here](https://docs.docker.com/storage/volumes/).

* Use mounted dataset with containerized toolkit
```
    docker run --gpus all -v /ssd/my_atacworks_data:/data claraomics/atacworks \
        /AtacWorks/peak2bw.py \
        /data/HSC.80M.chr123.10mb.peaks.bed \
        /data/hg19.auto.sizes \
        --prefix=/data/out
```
## FAQ
1. Unexpected bus error, how to troubleshoot?
```
    ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
```
A. This happens when docker container does not have enough shared memory allocation for deep learning frameworks. Checkout this doc on shared memory allocation [here](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#setincshmem). You can run your docker by adding this flag `--shm-size`. Increase the memory to 1GB or higher.

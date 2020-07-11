# AtacWorks Dockerfile

All the AtacWorks SDK components can be executed inside a docker environment. For ease of use,
the AtacWorks team has created a public docker image that can be pulled to run the toolkit.

## Install Docker with GPU support

Docker now ships with native support for GPU. Please follow the instructions on the [nvidia-docker](https://github.com/nvidia/nvidia-docker/wiki/Installation-(Native-GPU-Support))
page to setup the docker framework correctly.
Verify that the setup above completed successfully and your system has access to [Docker Hub](https://hub.docker.com/r/claraomics/atacworks).

We provide two different images to run AtacWorks from:
    * claraomics/atacworks:latest - This image contains pre-installed atacworks toolkit. It can be used for 
      running training, denoising and evaluating workflows without the need of cloning the repository. Look at the section
      "Pre-installed AtacWorks" below for instructions to pull that image.
    * claraomics/atacworks:source-latest - This image clones atacworks repository and installs atacworks from the
      source. This image is useful for getting familiar with the source code, to run atacworks tutorial notebooks etc. Look at the
      section "AtacWorks from Source" below for instructions to pull that image.

### Pre-installed AtacWorks
Run the following command to launch a docker container that contains pre-installed latest version of AtacWorks in interactive mode.

```
    docker run -it --gpus all --shm-size 2G claraomics/atacworks:latest
```

If the above command doesn't run successfully, please stop and re-install docker or contact the
AtacWorks dev team on GitHub for more help.

### AtacWorks from Source
To pull the docker image that clones and installs atacworks from source, run the following command:
```
    docker run -it --gpus all --shm-size 2G claraomics/atacworks:source-latest
```

If the above command doesn't run successfully, please stop and re-install docker or contact the
AtacWorks dev team on GitHub for more help.

## Run custom workflow in docker
Before starting on custom workflows please refer to the tutorials available in the README of the repository. Those will
help you get familiarized with the features of the SDK.
Once you are comfortable with them, you can use the docker environment to run all of the commands with ease.

Remember to use the tag "latest" or "source-latest" depending on whether you need AtacWorks source code or not. By default,
`claraomics/atacworks` will pull the "latest" tag.

* Mount volumes to the container
Use the `-v` option in docker to mount volumes with your data. Official documentation for the options can be found [here](https://docs.docker.com/storage/volumes/).

* Use mounted dataset with containerized toolkit, example shown below
```
    docker run --gpus all --shm-size 2G -v /ssd/my_atacworks_data:/data claraomics/atacworks \
        atacworks denoise \
        --noisybw /data/noisy.bw \
        --genome /data/hg19.auto.sizes \
        --out_home /data/output \
        --distributed
```

## Run Docker in interactive mode with port forwarding for Jupyter notebooks
```
docker run -it --gpus all --shm-size 2G -p 8888:8888 claraomics/atacworks:source-latest
```
Note: Jupyter notebook will have to be started manually once inside the container. Below is an example command to launch the jupyter-lab.
```
jupyter-lab --ip 0.0.0.0 --allow-root
```
The above command will print out a URl that you can open in your browser.

## FAQ
1. Unexpected bus error, how to troubleshoot?
```
    ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
```
A. This happens when docker container does not have enough shared memory allocation for deep learning frameworks. Checkout this doc on shared memory allocation [here](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#setincshmem). You can run your docker by adding this flag `--shm-size`. Increase the memory to 1GB or higher.

2. How can I build my own custom Dockerfile ?
A. AtacWorks repository contains a [Dockerfile](https://github.com/clara-parabricks/AtacWorks/blob/master/Dockerfile) to allow users to build a custom docker image. To build your own docker image, enter the root directory of AtacWorks repository.
```
cd <path-to-atacworks-root-dir>
```
Make and save changes if any to the Dockerfile. The run the following command to build a docker image.
```
docker build . -t atacworks:custom
```

You can launch a docker container with this image using the commands in the sections above. Just replace the image "claraomics/atacworks" with "atacworks:custom" or any other name you choose to give.


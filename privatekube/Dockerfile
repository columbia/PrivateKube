###
### Container image to interact with PrivateKube, for instance in Kubeflow Pipelines components
### Includes the privacy claims wrappers and other privacy utils
### Available on DockerHub: https://hub.docker.com/repository/docker/privatekube/sdk
### See `examples/pipeline` for an example
###

FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu16.04

RUN apt-get update \ 
    && apt-get install -y --no-install-recommends gcc python3-dev curl wget git

RUN curl -sSL https://sdk.cloud.google.com | bash
ENV PATH $PATH:/root/google-cloud-sdk/bin

# Conda environment
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/root/miniconda3/bin:$PATH"
RUN mkdir /root/.conda && bash Miniconda3-latest-Linux-x86_64.sh -b
RUN conda init bash \
    && . ~/.bashrc \
    && conda create --name privatekube python=3.8 \
    && conda activate privatekube


# Install PrivateKube and ML dependencies
RUN python -m pip install torchcsprng==0.1.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN python -m pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN git clone --single-branch --branch main https://github.com/columbia/PrivateKube.git
RUN python -m pip install -r PrivateKube/privatekube/requirements.txt
RUN python -m pip install -e PrivateKube/privatekube/

# Embeddings for TorchText
# (if they are not available, the embeddings will be downloaded when you run the evaluation macrobenchmark models)
COPY .vector_cache PrivateKube/evaluation/macrobenchmark/data/.vector_cache

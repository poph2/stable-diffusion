FROM nvidia/cuda:12.1.0-base-ubuntu20.04
# FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=nointeractive

# WORKDIR /stable-diffusion
# COPY . /stable-diffusionle-diffusion
# ENV PYTHONPATH "${PYTHONPATH}:/stable-difusion"

# RUN apt-get update -y
## Set up python & pip
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y python3-pip  build-essential git libglib2.0-0 libsm6 libxext6 libxrender-dev libfontconfig1


# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh
# RUN bash Miniconda3-py38_4.12.0-Linux-x86_64.sh -b -p $HOME/miniconda





## Install dependencies
#RUN yum install -y gcc gcc-c++
#RUN pip config set global.extra-index-url ${PIP_EXTRA_INDEX_URL}
#
#RUN pip install -r requirements.txt --no-cache-dir
#
## Download Divinia Models
#RUN python3 scripts/setup_models.py
#RUN mv /root/.divinia /data_eng
#RUN mv /root/nltk_data /data_eng

# RUN ls -la

# Command can be overwritten by providing a different command in the template directly.
#CMD ["lambdas.handler.hello"]


#aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 894841989444.dkr.ecr.us-east-1.amazonaws.com && docker build --progress=plain -f Dockerfile -t "894841989444.dkr.ecr.us-east-1.amazonaws.com/dai-data-eng:latest" . && docker push "894841989444.dkr.ecr.us-east-1.amazonaws.com/dai-data-eng:latest"


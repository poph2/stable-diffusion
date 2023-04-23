FROM stable-diffusion-base

WORKDIR /stable-diffusion
COPY . /stable-diffusion
ENV PYTHONPATH "${PYTHONPATH}:/stable-difusion"


RUN pip install -r requirements.txt


# python3 scripts/txt2img.py --prompt "a photorealistic vaporwave image of a lizard riding a snowboard through space" --plms --ckpt sd-v1-4.ckpt --skip_grid --n_samples 1



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


# hash:sha256:83577dde77a14bc4016eb2676993e2422d8dee1db2594ea23a73920bea422603
FROM registry.hackathon2023.radr-ai.com/codeocean/py-r:python3.10.12-R4.2.3-IRkernel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        default-jre=2:1.11-72build2 \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y \
        h2o-py==3.44.0.2 \
        openjdk==8.0.382 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    h2o==3.44.0.2

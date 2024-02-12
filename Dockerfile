FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install linux packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install python stuff
RUN apt-get --fix-missing update && \
    apt-get -y --no-install-recommends install \
        python3.10 \
        python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Setup default pip and python
RUN rm -f /usr/bin/pip && \
    ln -s /usr/bin/pip3 /usr/bin/pip && \
    rm -f /usr/bin/python && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    rm -f /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    pip install --no-cache-dir --upgrade pip

WORKDIR /

# Installing package requirements and latest version of speechbrain
COPY requirements.txt requirements.txt
ENV PIP_INDEX_URL https://repos.tech.orange/artifactory/api/pypi/pythonproxy/simple
RUN pip install --no-cache-dir --requirement requirements.txt
# Installing SpeechBrain develop branch
RUN git clone https://github.com/speechbrain/speechbrain.git 
WORKDIR /speechbrain 
RUN pip install --no-cache-dir --requirement requirements.txt && \
    pip install --no-cache-dir --editable .
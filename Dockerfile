# `docker build -t trefide --target trefide .`
FROM conda/miniconda3 AS trefide
WORKDIR /

RUN apt-get -y update && apt-get -y install wget \
                                            gnupg \
                                            apt-transport-https \
                                            ca-certificates \
                                            make \
                                            less \
                                            procps \
                                            cpio \
                                            g++

# opencv required system library
# https://github.com/conda-forge/pygridgen-feedstock/issues/10
RUN apt-get -y install libgl1-mesa-glx 

# Intel MKL installation: 
# https://software.intel.com/en-us/articles/installing-intel-free-libs-and-python-apt-repo
# https://gist.github.com/pachamaltese/afc4faef2f191b533556f261a46b3aa8
RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB 
RUN apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
RUN echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list
RUN apt-get -y update && apt-get -y install intel-mkl-64bit-2019.4-070 \
                                            intel-mkl-2019.4-070

COPY setup.py ./setup.py
COPY src ./src
COPY trefide ./trefide

ENV TREFIDE /
ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:$TREFIDE/src"
ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:$TREFIDE/src/proxtv"
ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:$TREFIDE/src/glmgen/lib"

WORKDIR src

# TODO - Unclear if mklvars.sh is necessary
RUN MKLVARS_ARCHITECTURE=intel64 /opt/intel/mkl/bin/mklvars.sh && make

WORKDIR /
RUN conda create -n trefide python=3.6 trefide -c jw3132 -c intel
RUN /bin/bash -c "source activate trefide && python setup.py install"

RUN /bin/bash -c "conda install -n trefide py-opencv mkl h5py"

COPY jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py

COPY demos ./demos

# TODO avoid module import issues
COPY run_pmd.py /usr/local/envs/trefide/lib/python3.6/site-packages/run_pmd.py
# run_readwrite just tests scipy.io.savemat
#COPY run_readwrite.py /usr/local/envs/trefide/lib/python3.6/site-packages/run_readwrite.py

#CMD "source activate trefide && source /opt/intel/mkl/bin/mklvars.sh intel64 && jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root --debug"
CMD /bin/bash -c "source activate trefide && jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root --debug"

#############################################
#############################################
FROM trefide AS funimag

# TODO - brute force searching for system library dependencies...
RUN apt-get -y update && apt-get -y install gcc \
																						libglib2.0-0 \
																						libsm6 \
																						libxrender1 \
																						libfontconfig1 \
																						libxext6 \
                                            vim \
                                            git

# Cache-buster; ADD for content from remote source, always retrieved and compared to cached version
# https://github.com/moby/moby/issues/14704
ADD https://api.github.com/repos/nik-sm/funimag/compare/master...HEAD /dev/null
RUN git clone https://github.com/nik-sm/funimag.git

WORKDIR funimag

EXPOSE 8888
RUN /bin/bash -c "source activate trefide && pip install -r requirements.txt"
RUN /bin/bash -c "source activate trefide && pip install -e ."
CMD /bin/bash -c "source activate trefide && jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --port=8888 --debug"

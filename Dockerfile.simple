# usage: 
# docker build -t trefide .
FROM conda/miniconda3 AS trefide
RUN conda create -n trefide python=3.6 trefide -c ikinsella -c intel -c menpo
RUN /bin/bash -c "source activate trefide && conda install h5py"
RUN /bin/bash -c "source activate trefide && python -c 'from trefide.pmd import batch_decompose, batch_recompose'"
RUN apt-get -y update && apt-get -y install libgtk2.0-dev
CMD /bin/bash -c "source activate trefide && jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --port=8888 --debug"

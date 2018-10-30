FROM python:2.7-slim

ENV RASA_NLU_DOCKER="YES" \
    RASA_NLU_HOME=/app \
    RASA_NLU_PYTHON_PACKAGES=/usr/local/lib/python2.7/dist-packages

VOLUME ["${RASA_NLU_HOME}", "${RASA_NLU_PYTHON_PACKAGES}"]

WORKDIR ${RASA_NLU_HOME}/rasa_nlu

# Run updates, install basics and cleanup
# - build-essential: Compile specific dependencies
# - git-core: Checkout git repos
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
  build-essential \
  git-core && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# and fixing problems occuring due to package being not backwards compatible
RUN pip install spacy==1.2.0
RUN pip install scikit-learn
RUN pip install pathlib \
    cloudpickle \
    boto3 \
    pytest-pep8 \
    requests \
    progressbar

# manually installing MITIE
# The Github at tag 0.2 missed a lib folder and a setup.py file. Copied these from tag 0.3 and 0.5 respectively.
COPY ./MITIE ${RASA_NLU_HOME}/MITIE
RUN pip install -e ${RASA_NLU_HOME}/MITIE

# RUN ln -s ${RASA_NLU_HOME}/rasa_nlu/setup.py setup.py
# RUN python setup.py install
# RUN pip install -e ${RASA_NLU_HOME}/rasa_nlu
RUN pip install rasa_nlu==0.5

EXPOSE 5000

RUN git clone --branch 0.5 --depth 1 https://github.com/RasaHQ/rasa_nlu ${RASA_NLU_HOME}/rasa_nlu
RUN ln -s ${RASA_NLU_HOME}/rasa_nlu/entrypoint.sh ./entrypoint.sh

# ENTRYPOINT ["./entrypoint.sh"]
# CMD ["help"]

# To browse Docker run -i -t --entrypoint /bin/bash <image ID>
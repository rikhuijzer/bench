FROM python:3.6-slim

ENV RASA_NLU_DOCKER="YES" \
    RASA_NLU_HOME=/app \
    RASA_NLU_PYTHON_PACKAGES=/usr/local/lib/python3.6/dist-packages

# Run updates, install basics and cleanup
# - build-essential: Compile specific dependencies
# - git-core: Checkout git repos
RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends build-essential git-core openssl libssl-dev libffi6 libffi-dev curl  \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR ${RASA_NLU_HOME}

# use bash always
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN git clone --branch 0.13.7 --depth 1 https://github.com/RasaHQ/rasa_nlu ${RASA_NLU_HOME}/rasa_nlu

RUN pip install -r ${RASA_NLU_HOME}/rasa_nlu/alt_requirements/requirements_spacy_sklearn.txt

RUN pip install -e ${RASA_NLU_HOME}/rasa_nlu

RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-2.0.0/en_core_web_md-2.0.0.tar.gz --no-cache-dir > /dev/null \
    && python -m spacy link en_core_web_md en \
    && pip install https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-2.0.0/de_core_news_sm-2.0.0.tar.gz --no-cache-dir > /dev/null \
    && python -m spacy link de_core_news_sm de

RUN ln -s ${RASA_NLU_HOME}/rasa_nlu/sample_configs/config_spacy.yml ${RASA_NLU_HOME}/config.yml
RUN ln -s ${RASA_NLU_HOME}/rasa_nlu/entrypoint.sh ./entrypoint.sh

VOLUME ["/app/projects", "/app/logs", "/app/data"]

EXPOSE 5000

ENTRYPOINT ["./entrypoint.sh"]
CMD ["start", "-c", "config.yml", "--path", "/app/projects"]
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

## install java stuff

RUN echo "deb http://http.debian.net/debian jessie-backports main" > /etc/apt/sources.list.d/jessie-backports.list

RUN apt-get update && apt-get install -y --no-install-recommends \
		bzip2 \
		unzip \
		xz-utils \
	&& rm -rf /var/lib/apt/lists/*

# Default to UTF-8 file.encoding
ENV LANG C.UTF-8

# add a simple script that can auto-detect the appropriate JAVA_HOME value
# based on whether the JDK or only the JRE is installed
RUN { \
		echo '#!/bin/sh'; \
		echo 'set -e'; \
		echo; \
		echo 'dirname "$(dirname "$(readlink -f "$(which javac || which java)")")"'; \
	} > /usr/local/bin/docker-java-home \
	&& chmod +x /usr/local/bin/docker-java-home

# do some fancy footwork to create a JAVA_HOME that's cross-architecture-safe
RUN ln -svT "/usr/lib/jvm/java-8-openjdk-$(dpkg --print-architecture)" /docker-java-home
ENV JAVA_HOME /docker-java-home

ENV JAVA_VERSION 8u141
ENV JAVA_DEBIAN_VERSION 8u141-b15-1~deb9u1

# see https://bugs.debian.org/775775
# and https://github.com/docker-library/java/issues/19#issuecomment-70546872
ENV CA_CERTIFICATES_JAVA_VERSION 20170531+nmu1

# added openjdk-8-jre-headless (source: https://lists.debian.org/debian-user/2017/09/msg00629.html)
RUN set -ex; \
# deal with slim variants not having man page directories (which causes "update-alternatives" to fail)
	if [ ! -d /usr/share/man/man1 ]; then \
		mkdir -p /usr/share/man/man1; \
	fi; \
	apt-get update; \
	apt-get install -y openjdk-8-jdk \
		ca-certificates-java \
	; \
	rm -rf /var/lib/apt/lists/*; \
# verify that "docker-java-home" returns what we expect
	[ "$(readlink -f "$JAVA_HOME")" = "$(docker-java-home)" ]; \
# update-alternatives so that future installs of other OpenJDK versions don't change /usr/bin/java
	update-alternatives --get-selections | awk -v home="$(readlink -f "$JAVA_HOME")" 'index($3, home) == 1 { $2 = "manual"; print | "update-alternatives --set-selections" }'; \
# ... and verify that it actually worked for one of the alternatives we care about
	update-alternatives --query java | grep -q 'Status: manual'

# see CA_CERTIFICATES_JAVA_VERSION notes above
RUN /var/lib/dpkg/info/ca-certificates-java.postinst configure

## done java

# this is taken from much earlier, but can be done later to avoid rebuilding all Docker layers constantly
# COPY . ${RASA_NLU_HOME}

RUN git clone --branch 0.13.7 --depth 1 https://github.com/RasaHQ/rasa_nlu ${RASA_NLU_HOME}/rasa_nlu

RUN pip install -r ${RASA_NLU_HOME}/rasa_nlu/alt_requirements/requirements_full.txt

RUN pip install -e ${RASA_NLU_HOME}/rasa_nlu

RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends wget \
    && wget -P data/ https://s3-eu-west-1.amazonaws.com/mitie/total_word_feature_extractor.dat \
    && apt-get remove -y wget \
    && apt-get autoremove -y

RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-2.0.0/en_core_web_md-2.0.0.tar.gz --no-cache-dir > /dev/null \
    && python -m spacy link en_core_web_md en \
    && pip install https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-2.0.0/de_core_news_sm-2.0.0.tar.gz --no-cache-dir > /dev/null \
    && python -m spacy link de_core_news_sm de

# COPY ${RASA_NLU_HOME}/rasa_nlu/sample_configs/config_spacy_duckling.yml ${RASA_NLU_HOME}/config.yml
RUN ln -s ${RASA_NLU_HOME}/rasa_nlu/sample_configs/config_spacy_duckling.yml ${RASA_NLU_HOME}/config.yml
#VOLUME ["/app/projects", "/app/logs", "/app/data"]

# RUN wget https://github.com/RasaHQ/rasa_nlu/blob/0.13.7/sample_configs/config_spacy_duckling.yml
# COPY config_spacy_duckling.yml ${RASA_NLU_HOME}/config.yml

# remove copy from top to avoid rebuild
EXPOSE 5000

ENTRYPOINT ["./entrypoint.sh"]
CMD ["start", "-c", "config.yml", "--path", "/app/projects"]
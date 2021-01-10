FROM ubuntu:20.04

# needed by Pipenv
ENV DEBIAN_FRONTEND noninteractive
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ENV PATH="/usr/local/bin:${PATH}"

RUN apt-get update && apt-get upgrade -y

RUN apt-get install -y                                                       \
       build-essential                                                       \
       software-properties-common                                            \
       vim                                                                   \
       make                                                                  \
       git                                                                   \
       less                                                                  \
       curl                                                                  \
       wget                                                                  \
       zlib1g-dev                                                            \
       libssl-dev                                                            \
       libffi-dev                                                            \
       python3-venv                                                          \
       python3-pip                                                           \
       python3-dev

# matplotlib build dependencies
RUN apt-get install -y                                                       \
       libxft-dev                                                            \
       libfreetype6                                                          \
       libfreetype6-dev

RUN mkdir /build
WORKDIR /build
COPY . /build

RUN pip3 install pipenv
RUN pipenv --python python3.8
RUN pipenv install --dev --skip-lock

CMD ["/bin/bash"]

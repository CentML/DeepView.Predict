FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04
MAINTAINER Geoffrey Yu <gxyu@cs.toronto.edu>
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata

RUN apt-get update --fix-missing && \
  apt-get install --no-install-recommends -y software-properties-common && \
  apt-get update && \
  apt-get install --no-install-recommends -y sudo && \
  apt-get install --no-install-recommends -y python3-pip python3-setuptools python3-dev && \
  apt-get install --no-install-recommends -y wget bzip2 ca-certificates libssl-dev && \
  rm -rf /var/lib/apt/lists/*

RUN pip3 install wheel && pip3 install numpy PyYAML
RUN pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
RUN pip3 install pillow>=7.2.0 pandas>=1.1.2 tqdm>=4.49.0

# Download cmake
# RUN wget "https://github.com/Kitware/CMake/releases/download/v3.17.0-rc1/cmake-3.17.0-rc1.tar.gz" -O /opt/cmake-3.17.0-rc1.tar.gz && \
#  cd /opt && tar xzf cmake-3.17.0-rc1.tar.gz

# Install cmake
#RUN cd /opt/cmake-3.17.0-rc1 && \
#  ./bootstrap && \
#  make -j && \
#  make install

#iInstall latest cmake
ADD https://github.com/Kitware/CMake/releases/download/v3.24.0/cmake-3.24.0-linux-x86_64.sh /cmake-3.24.0-linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.24.0-linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

# NOTE: gosu is used in create-user.sh
RUN mkdir ~/.gnupg && echo "disable-ipv6" >> ~/.gnupg/dirmngr.conf
RUN gpg --keyserver keyserver.ubuntu.com --recv-keys B42F6819007F00F88E364FD4036A9C25BF357DD4
RUN wget "https://github.com/tianon/gosu/releases/download/1.11/gosu-$(dpkg --print-architecture | awk -F- '{ print $NF }')" -O /usr/local/bin/gosu && \
  wget "https://github.com/tianon/gosu/releases/download/1.11/gosu-$(dpkg --print-architecture | awk -F- '{ print $NF }').asc" -O /usr/local/bin/gosu.asc && \
  gpg --verify /usr/local/bin/gosu.asc && \
  rm /usr/local/bin/gosu.asc && \
  chmod +x /usr/local/bin/gosu

COPY create-user.sh /usr/local/bin/create-user.sh
RUN chmod +x /usr/local/bin/create-user.sh
ENTRYPOINT ["/usr/local/bin/create-user.sh"]
CMD ["/bin/bash"]

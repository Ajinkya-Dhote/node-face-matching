FROM tensorflow/tensorflow:latest
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - &&  apt-get install -y nodejs


WORKDIR /usr/app

COPY package.json .
RUN npm install
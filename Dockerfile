# Build Stage
FROM node:18 as build-stage

RUN apt-get update && apt-get install -y build-essential python3
RUN npm set strict-ssl false
RUN npm install -g npm@latest

# Create app directory
RUN mkdir -p /app
WORKDIR /app

# Install app dependencies
COPY package.json .
COPY package-lock.json .
RUN npm install
RUN npm rebuild @tensorflow/tfjs-node --build-from-source
RUN npm cache clean --force

# Application stage
FROM node:18
WORKDIR /app

# Install necessary libraries for TensorFlow.js
RUN apt-get update && apt-get install -y libstdc++6 libgcc1 libc6

# Copy node modules and app code
COPY --from=build-stage /app/node_modules ./node_modules
COPY . .

# User configuration directory volume
VOLUME ["/data"]
EXPOSE 3000

CMD ["node", "index.js"]

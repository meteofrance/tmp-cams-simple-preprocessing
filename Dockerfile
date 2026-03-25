# Use a python image with python 3.12 and pytorch 2.10 pre-installed
FROM python:3.12-slim

# Install the project into `/app`
WORKDIR /app

# Install the necessary libraries
RUN apt-get update \
    && apt-get install -y curl gcc g++ sudo git wget \
    && apt-get install -y libgeos-dev libeccodes-dev libeccodes-tools

# Install project dependencies
COPY pyproject.toml pyproject.toml
RUN pip install .
# specifies the base image for the Docker COntainer 
# slim: an official DOcker image. Lighter, more efficient
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    gcc \
    g++ \
    make \
    tesseract-ocr \
    pkg-config \
    libhdf5-dev \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
# all commands that follow will be executed in this directory
WORKDIR /app

# copy over all desired folders.
COPY data/ctpn.pb /app/data/ctpn.pb
COPY lib /app/lib
COPY models/trained /app/models/trained
COPY scripts /app/scripts
COPY app.py /app/app.py
COPY requirements.txt /app/requirements.txt

# install all required packages 
# prevent caching of installed packages to reduce the image size
RUN pip install --no-cache-dir -r requirements.txt 

# indicates which port the application wil use to users and tools
EXPOSE 5000

# define environment variables
ENV NAME World
ENV CTPN_PATH /app/data/ctpn.pb
ENV IDENTIFICATION_MODEL_PATH /app/models/trained

# this runs when the container starts. Executes python app.py
CMD ["python", "app.py"]
# Simple Solution

FROM python:3.10.6-buster

WORKDIR /prod

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY taxifare taxifare
COPY setup.py setup.py
RUN pip install .

COPY Makefile Makefile
RUN make reset_local_files

CMD uvicorn taxifare.api.fast:app --host 0.0.0.0 --port $PORT


# Optimised Solution

# tensorflow base-images are optimized: lighter than python-buster + pip install tensorflow
# FROM tensorflow/tensorflow:2.10.0

# WORKDIR /prod

# # We strip the requirements from useless packages like `ipykernel`, `matplotlib` etc...
# COPY requirements_prod.txt requirements.txt
# RUN pip install -r requirements.txt

# COPY taxifare taxifare
# COPY setup.py setup.py
# RUN pip install .

# COPY Makefile Makefile
# RUN make reset_local_files

# CMD uvicorn taxifare.api.fast:app --host 0.0.0.0 --port $PORT

FROM python:3.12-slim

# This is only necessary when installing packages from github

# RUN apt-get update \
#  && apt-get install -y --no-install-recommends \
#     curl=7.88.1-10+deb12u7 \
#     git=1:2.39.5-0+deb12u1 \
#  && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ./jane_the_llama ./jane_the_llama

CMD [ "fastapi", "run", "./jane_the_llama/api.py" ]

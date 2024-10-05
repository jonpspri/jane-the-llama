FROM python:3.12-slim

WORKDIR /usr/src/app
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    curl=7.88.1-10+deb12u7 \
    git=1:2.39.5-0+deb12u1 \
 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "echo", "You must explicitly start each process" ]

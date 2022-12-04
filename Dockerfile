FROM python:3.10-bullseye

WORKDIR /project

RUN apt-get update && apt-get install -y \
  ffmpeg \
&& apt-get -y autoremove && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

CMD ["make"]

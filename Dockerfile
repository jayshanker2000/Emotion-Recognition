FROM ubuntu:20.04

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# Set dependencies
RUN apt -y update && apt-get -y upgrade
RUN apt install -y python3-pip
RUN apt install -y dos2unix

# Set working directory and copy files
WORKDIR /usr/app/src
COPY . .
RUN ls
RUN pip3 install -r requirements.txt

ENV FLASK_APP=app.py

EXPOSE 5000

# Run entrypoint file
RUN chmod +x entrypoint.sh
RUN dos2unix entrypoint.sh

ENTRYPOINT ["/usr/app/src/entrypoint.sh"]
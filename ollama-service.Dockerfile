FROM ollama/ollama:latest

RUN apt-get update && apt-get install -y curl

WORKDIR /usr/bin

COPY start.sh .
RUN chmod +x start.sh

ENTRYPOINT ["./start.sh"]

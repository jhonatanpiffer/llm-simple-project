#!/bin/bash

# Inicia o serviço do Ollama em segundo plano
ollama serve &

# Espera até que o servidor Ollama esteja de pé
echo "Aguardando o servidor Ollama iniciar..."
while ! curl --silent --output /dev/null http://localhost:11434/api/tags; do
    sleep 5
done
echo "Servidor Ollama está pronto."

# Baixa o modelo especificado
echo "Baixando o modelo tinyllama..."
#ollama pull tinyllama
ollama pull llama3.2

# Mantém o processo principal do Ollama em execução em primeiro plano
wait

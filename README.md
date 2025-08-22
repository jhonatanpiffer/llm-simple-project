# Desafio de Engenharia de IA: API de Extração de Dados de Incidentes
 Este projeto implementa uma API em Python para extrair informações estruturadas de descrições textuais de incidentes, utilizando um modelo de linguagem (LLM) local rodando via Ollama. A solução foi projetada com foco em boas práticas de desenvolvimento, testabilidade e reprodutibilidade, conforme os critérios do desafio.

## Estrutura do Projeto
A solução é composta pelos seguintes arquivos:

- `main.py`: O core da aplicação FastAPI, com o endpoint de extração, o pipeline de pré-processamento e a lógica de interação com o Ollama.

- `models.py`: Define os modelos de dados de entrada e saída da API usando Pydantic, garantindo validação e documentação automáticas.

- `requirements.txt`: Lista todas as dependências do Python.

- `tests/test_main.py`: Contém os testes unitários da API, usando pytest e unittest.mock para simular o comportamento do LLM.

- `Dockerfile`: Instruções para empacotar a API em um contêiner Docker, garantindo um ambiente de execução isolado.

- `ollama-service.Dockerfile`: Um Dockerfile dedicado para o serviço Ollama, que garante que o modelo seja baixado e o serviço esteja pronto antes da API iniciar.

- `start.sh`: Um script de shell que orquestra a inicialização do servidor Ollama e o download do modelo, resolvendo problemas de dependência e tempo de inicialização.

- `docker-compose.yaml`: Orquestra a execução de todos os contêineres do projeto: a API (fastapi_api) e o servidor do LLM (ollama_service).

- `.env`: Armazena variáveis de ambiente como o endereço do Ollama e o nome do modelo.

## Pré-requisitos
Para rodar o projeto, você precisa ter o Docker e o Docker Compose instalados em sua máquina.

## Instruções de Execução
Siga os passos abaixo para configurar e rodar a API.

### Passo 1: Rodar a Aplicação com Docker Compose
O docker-compose.yaml irá construir as imagens, iniciar os servidores, baixar o modelo e gerenciar a comunicação entre eles. A execução é feita com um único comando:

```Bash```

```docker compose up --build```

Atenção: A primeira execução pode levar alguns minutos, pois irá baixar a imagem do Ollama e o modelo tinyllama (cerca de 2,4GB). O script start.sh irá gerenciar o download de forma automatizada.

### Passo 2: Acessar a API
A API estará disponível em http://localhost:8000. Você pode verificar a documentação interativa (Swagger UI) em http://localhost:8000/docs e o ReDoc em http://localhost:8000/redoc.

### Passo 3: Testar o Endpoint
Você pode enviar uma requisição POST para o endpoint /extract usando o curl.

Exemplo de requisição:

```Bash```

```
 curl -X 'POST' \
  'http://localhost:8000/extract' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "Em 14 de agosto de 2025, no datacenter central, uma queda de energia afetou a rede interna por 3 horas."
}'
``` 
Saída esperada:

``` JSON``` 
``` 
{
  "data_ocorrencia": "2025-08-14",
  "local": "datacenter central",
  "tipo_incidente": "Queda de energia",
  "impacto": "Rede interna indisponível por 3 horas"
}
``` 
## Rodando os Testes Unitários
Para rodar os testes, use o pytest. Os testes foram projetados para rodar de forma isolada, sem depender do servidor Ollama, usando a técnica de mocking.

``` Bash``` 

``` pytest``` 

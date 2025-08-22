import os
import json
import logging
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError
from normality import normalize, collapse_spaces
from dotenv import load_dotenv

from models import IncidentInput, IncidentOutput

# Carrega as variáveis de ambiente do arquivo.env
load_dotenv()

# Configura o logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Instancia a aplicação FastAPI
app = FastAPI(
    title="API de Extração de Dados de Incidentes com LLM Local",
    description="Uma API que extrai informações estruturadas de descrições de incidentes usando um LLM local (Ollama).",
)

# Configurações do LLM
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "tinyllama")

# Cliente HTTP assíncrono para comunicação com o Ollama
ollama_client = httpx.AsyncClient(base_url=OLLAMA_HOST, timeout=600.0)

def preprocess_text(text: str) -> str:
    """
    Implementa um pipeline simples de pré-processamento de texto.
    """
    # Normaliza o texto, convertendo para minúsculas e limpando espaços/diacríticos.[7]
    normalized_text = normalize(text)
    return collapse_spaces(normalized_text)

@app.get("/")
async def read_root():
    return {"message": "Bem-vindo à API de Extração de Incidentes!"}

@app.post(
    "/extract",
    summary="Extrai dados de um incidente",
    response_model=IncidentOutput,
    tags=["extração"],
)
async def extract_incident_data(input_data: IncidentInput):
    """
    Processa uma descrição de incidente, extrai dados e retorna um JSON.
    """
    logger.info("Iniciando a extração de dados para o incidente.")

    # Passo 1: Pré-processamento do texto
    processed_text = preprocess_text(input_data.text)
    logger.info(f"Texto pré-processado: {processed_text}")

    # Passo 2: Prompt Engineering
    # Define o schema JSON esperado a partir do modelo Pydantic.
    json_schema = IncidentOutput.model_json_schema()
    
    # Exemplo "few-shot" do próprio desafio.[10, 1]
    example_input = "Ontem às 14h, no escritório de São Paulo, houve uma falha no servidor principal que afetou o sistema de faturamento por 2 horas."
    example_output = {
        "data_ocorrencia": "2025-08-12 14:00",
        "local": "São Paulo",
        "tipo_incidente": "Falha no servidor",
        "impacto": "Sistema de faturamento indisponível por 2 horas"
    }

    prompt = f"""
    Extract the following information from the incident text below.
    Return ONLY a valid JSON object, without any additional text or explanations.

    Your response must strictly follow the following JSON schema:
    {json.dumps(json_schema, indent=2)}

    Example of input and output:
    Input: "{example_input}"
    Output: {json.dumps(example_output, indent=2)}

    Text for analysis: "{processed_text}"
    """
    
    # Passo 3: Chamada ao LLM Local via Ollama
    try:
        response = await ollama_client.post(
            "/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,  # Desabilita o streaming para receber a resposta completa de uma vez.[9]
            },
        )
        response.raise_for_status()
        raw_output = response.json().get("response", "").strip()
        
        logger.info(f"Resposta bruta do Ollama: {raw_output}")

        # Remove o bloco de código Markdown se presente
        if raw_output.startswith("```json") and raw_output.endswith("```"):
            raw_output = raw_output[7:-3].strip()

    except httpx.HTTPStatusError as e:
        logger.error(f"Erro na comunicação com o Ollama: {e.response.text}")
        raise HTTPException(
            status_code=500, detail=f"Erro na comunicação com o serviço LLM: {e.response.text}"
        )
    except Exception as e:
        logger.error(f"Erro inesperado: {e}")
        raise HTTPException(status_code=500, detail="Erro inesperado ao processar a requisição.")
    
    # Passo 4: Validação e retorno do JSON
    try:
        llm_output = json.loads(raw_output)
        
        # O Pydantic valida e serializa o JSON automaticamente.[4, 5]
        return IncidentOutput.model_validate(llm_output)

    except json.JSONDecodeError as e:
        logger.error(f"A resposta do LLM não é um JSON válido: {e}")
        raise HTTPException(
            status_code=500, detail="O LLM retornou um formato inválido. Não é um JSON."
        )
    except ValidationError as e:
        logger.error(f"A resposta do LLM não corresponde ao schema Pydantic: {e}")
        raise HTTPException(
            status_code=500, detail="O LLM retornou um JSON malformado. Não corresponde ao schema esperado."
        )

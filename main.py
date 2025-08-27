import os
import logging
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from normality import normalize, collapse_spaces
import dspy
from dspy.teleprompt import BootstrapFewShot

from models import IncidentInput, IncidentOutput

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Configura o logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Instancia a aplicação FastAPI
app = FastAPI(
    title="API de Extração de Dados de Incidentes com DSPy e LLM Local",
    description="Uma API que extrai informações estruturadas de descrições de incidentes usando DSPy e um LLM local (Ollama).",
)

# --- Configuração do DSPy ---

# Configurações do LLM a partir das variáveis de ambiente
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2")

try:
    lm = dspy.LM(f"ollama_chat/{MODEL_NAME}", api_base=OLLAMA_HOST)
    dspy.configure(lm=lm)

    logger.info(f"DSPy configurado com sucesso para o modelo: {MODEL_NAME} no host: {OLLAMA_HOST}")
except Exception as e:
    logger.error(f"Falha ao configurar o dspy.Ollama: {e}")
    # Em um cenário de produção, você pode querer encerrar a aplicação aqui.
    ollama_lm = None

# Define a "assinatura" (Signature) para a tarefa de extração de informações.
class ExtractIncidentSignature(dspy.Signature):
    """Extrai informações estruturadas de um texto de incidente."""

    text: str = dspy.InputField(desc="Textual description of the incident.")
    date: str = dspy.OutputField(desc="Date and time of the incident")
    local: str = dspy.OutputField(desc="Location of the incident")
    tipo: str = dspy.OutputField(desc="Type or category of the incident")
    impacto: str = dspy.OutputField(desc="Brief description of the generated impact")
# --- Otimização do Módulo DSPy (Compilação Única) ---

# Cria um módulo "Predict" do DSPy com a assinatura que definimos.
incident_extractor = dspy.Predict(ExtractIncidentSignature)

# Exemplo de "few-shot" para guiar o modelo
example_input = "Ontem às 14h, no escritório de São Paulo, houve uma falha no servidor principal que afetou o sistema de faturamento por 2 horas."
train_example = dspy.Example(
    text=example_input,
    date="Ontem às 14h",
    local="escritório de São Paulo",
    tipo="falha no servidor principal",
    impacto="afetou o sistema de faturamento por 2 horas"
).with_inputs("text")

# Compila o otimizador (teleprompter) uma única vez na inicialização da aplicação
teleprompter = BootstrapFewShot()
optimized_incident_extractor = teleprompter.compile(incident_extractor, trainset=[train_example])

logger.info("Módulo DSPy compilado e otimizado com sucesso.")

# --- Fim da Configuração e Otimização ---

def preprocess_text(text: str) -> str:
    """
    Implementa um pipeline simples de pré-processamento de texto.
    """
    #normalized_text = normalize(text)
    return collapse_spaces(text)

@app.get("/")
async def read_root():
    return {"message": "Bem-vindo à API de Extração de Incidentes com DSPy!"}

@app.post(
    "/extract",
    summary="Extrai dados de um incidente usando DSPy",
    response_model=IncidentOutput,
    tags=["extração"],
)
async def extract_incident_data(input_data: IncidentInput):
    """
    Processa uma descrição de incidente, extrai dados com DSPy e retorna um JSON.
    """
    logger.info("Iniciando a extração de dados para o incidente com DSPy.")

    processed_text = preprocess_text(input_data.text)
    logger.info(f"Texto pré-processado: {processed_text}")

    try:
        # Executa a extração usando o módulo já otimizado
        prediction = optimized_incident_extractor(text=processed_text)

        logger.info(f"Resposta estruturada do DSPy: {prediction}")

        # Validação e retorno com Pydantic
        return IncidentOutput(
            date=prediction.date,
            local=prediction.local,
            tipo=prediction.tipo,
            impacto=prediction.impacto,
        )

    except Exception as e:
        logger.error(f"Erro durante a predição do DSPy ou validação do Pydantic: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro inesperado ao processar a requisição com o LLM: {str(e)}"
        )


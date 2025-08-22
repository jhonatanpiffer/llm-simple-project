import pytest
from fastapi.testclient import TestClient
from main import app, ollama_client
from unittest.mock import AsyncMock, patch

client = TestClient(app)

# Cenário 1: Resposta do LLM válida e completa
def mock_ollama_valid_response(mock_client):
    """
    Simula uma resposta válida e completa do Ollama.
    """
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "response": """
        {
            "data_ocorrencia": "2025-08-12 14:00",
            "local": "São Paulo",
            "tipo_incidente": "Falha no servidor",
            "impacto": "Sistema de faturamento indisponível por 2 horas"
        }
        """
    }
    mock_client.post.return_value = mock_response

# Cenário 2: Resposta do LLM com JSON malformado
def mock_ollama_invalid_json(mock_client):
    """
    Simula uma resposta do Ollama que não é um JSON válido.
    """
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "Isto não é um JSON válido."}
    mock_client.post.return_value = mock_response

# Cenário 3: Resposta do LLM com campos ausentes
def mock_ollama_incomplete_response(mock_client):
    """
    Simula uma resposta JSON válida, mas com campos ausentes.
    """
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "response": """
        {
            "local": "Brasília",
            "tipo_incidente": "Incidente de segurança"
        }
        """
    }
    mock_client.post.return_value = mock_response

# Cenário 4: Erro na comunicação com o Ollama
def mock_ollama_http_error(mock_client):
    """
    Simula um erro HTTP na chamada ao Ollama.
    """
    mock_client.post.side_effect = httpx.HTTPStatusError(
        message="503 Service Unavailable",
        request=httpx.Request("POST", "http://localhost:11434/api/generate"),
        response=httpx.Response(503, request=httpx.Request("POST", "url")),
    )

@pytest.mark.asyncio
async def test_extract_incident_data_success():
    """
    Testa se a API retorna a resposta esperada quando o LLM retorna um JSON válido.
    """
    with patch("main.ollama_client", new_callable=AsyncMock) as mock_client:
        mock_ollama_valid_response(mock_client)

        response = client.post(
            "/extract",
            json={"text": "Ontem às 14h, no escritório de São Paulo, houve uma falha no servidor principal que afetou o sistema de faturamento por 2 horas."}
        )

        assert response.status_code == 200
        assert response.json() == {
            "data_ocorrencia": "2025-08-12 14:00",
            "local": "São Paulo",
            "tipo_incidente": "Falha no servidor",
            "impacto": "Sistema de faturamento indisponível por 2 horas",
        }

@pytest.mark.asyncio
async def test_extract_incident_data_invalid_json_from_llm():
    """
    Testa se a API lida corretamente com uma resposta do LLM que não é JSON.
    """
    with patch("main.ollama_client", new_callable=AsyncMock) as mock_client:
        mock_ollama_invalid_json(mock_client)

        response = client.post(
            "/extract",
            json={"text": "Houve um incidente."}
        )

        assert response.status_code == 500
        assert "não é um JSON válido" in response.json()["detail"]

@pytest.mark.asyncio
async def test_extract_incident_data_incomplete_response():
    """
    Testa se a API lida com um JSON válido, mas com campos ausentes.
    """
    with patch("main.ollama_client", new_callable=AsyncMock) as mock_client:
        mock_ollama_incomplete_response(mock_client)

        response = client.post(
            "/extract",
            json={"text": "Houve um incidente de segurança em Brasília."}
        )

        assert response.status_code == 200
        assert response.json() == {
            "data_ocorrencia": None,
            "local": "Brasília",
            "tipo_incidente": "Incidente de segurança",
            "impacto": None,
        }

@pytest.mark.asyncio
async def test_extract_incident_data_http_error():
    """
    Testa se a API lida com erros de comunicação com o Ollama.
    """
    with patch("main.ollama_client", new_callable=AsyncMock) as mock_client:
        mock_ollama_http_error(mock_client)

        response = client.post(
            "/extract",
            json={"text": "Houve um incidente."}
        )

        assert response.status_code == 500
        assert "Erro na comunicação com o serviço LLM" in response.json()["detail"]

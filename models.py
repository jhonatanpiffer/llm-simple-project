from pydantic import BaseModel, Field
from typing import Optional

class IncidentInput(BaseModel):
    """
    Schema para o corpo da requisição de entrada.
    """
    text: str = Field(..., description="Descrição textual do incidente.")

class IncidentOutput(BaseModel):
    """
    Schema para a saída estruturada do incidente, conforme especificado no desafio.
    """
    date: Optional[str] = Field(None, description="Data e hora do incidente.")
    local: Optional[str] = Field(None, description="Local do incidente.")
    tipo: Optional[str] = Field(None, description="Tipo ou categoria do incidente.")
    impacto: Optional[str] = Field(None, description="Descrição breve do impacto gerado.")

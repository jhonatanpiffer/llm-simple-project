import dspy
lm = dspy.LM("ollama_chat/llama3.2", api_base="http://localhost:11434", api_key="")
dspy.configure(lm=lm)

class ExtractInfo(dspy.Signature):
    """Extract structured information from text."""

    text: str = dspy.InputField()
    title: str = dspy.OutputField()
    headings: list[str] = dspy.OutputField()
    entities: list[dict[str, str]] = dspy.OutputField(desc="a list of entities and their metadata")

class ExtractInfoIncident(dspy.Signature):
    """Extract structured information from text."""

    text: str = dspy.InputField()
    data_ocorrencia: str = dspy.OutputField(desc="data e hora do incidente (se presente no texto)")
    local: str = dspy.OutputField(desc="local do incidente")
    tipo_incidente: str = dspy.OutputField(desc="tipo ou categoria do incidente")
    impacto: str = dspy.OutputField(desc="descrição breve do impacto")
module = dspy.Predict(ExtractInfoIncident)

text = "Ontem às 14h, no escritório de São Paulo, houve uma falha no " \
    "servidor principal que afetou o sistema de faturamento por 2 " \
    "horas."
response = module(text=text)

print(response.data_ocorrencia)
print(response.local)
print(response.tipo_incidente)
print(response.impacto)

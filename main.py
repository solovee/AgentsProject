from datetime import datetime
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os   

load_dotenv()  # Carrega variáveis de ambiente do arquivo .env

# =========  MODELO LOCAL (Llama3 via Ollama) =========
system_prompt = """
Você é um agente inteligente que segue o formato ReAct (Raciocínio + Ação).

Use o formato abaixo para cada interação:

Thought: descreva brevemente o que você vai fazer.
Action: escolha UMA ferramenta (entre as disponíveis) para obter informações ou realizar cálculos.
Action Input: passe o input necessário para a ferramenta.

Após receber a resposta da ferramenta, pense novamente (Thought) e decida:
- Se ainda precisar buscar algo → use outra Action.
- Se já tiver todas as informações → finalize com:

Final Answer: [resposta completa ao usuário em português]

⚠️ Regras importantes:
- Nunca escreva "Action: None".
- Sempre que a resposta estiver pronta, finalize com "Final Answer:".
- Evite repetir "Thought:Thought:". Use apenas um "Thought:" por vez.
- Mantenha o raciocínio breve e direto.
- Se o usuário fizer mais de uma pergunta na mesma mensagem,
responda **todas elas**, usando a memória da conversa quando necessário.
"""



llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    api_key=os.getenv("API_KEY")
)




    # =========  FERRAMENTAS =========
search = DuckDuckGoSearchRun()

from langchain.prompts import ChatPromptTemplate

def traduz(expressao: str) -> str:
    try:
        # Cria um prompt personalizado apenas para este uso
        prompt = ChatPromptTemplate.from_template("""
        Você é um tradutor de idiomas.
        Traduza o texto para portugues brasileiro
        Texto: {texto}
        """)

        # Monta a cadeia (prompt → llm)
        chain = prompt | llm  # isso combina o template com o modelo

        # Executa apenas esse fluxo
        resposta = chain.invoke({"texto": expressao})
        return resposta.content

    except Exception as e:
        return f"Erro ao traduzir: {e}"

def calcular(expressao: str) -> str:
    try:
        resultado = eval(expressao)
        return f"O resultado de {expressao} é {resultado}"
    except Exception as e:
        return f"Erro ao calcular: {e}"

def dataHoje(*args, **kwargs) -> str:
    try:
        data_hoje = datetime.now().strftime("%d/%m/%Y")
        return data_hoje
    except Exception as e:
        return f"Erro ao pegar data: {e}"
    
tools = [
    Tool(
        name="BuscaWeb",
        func=search.run,
        description="Use para buscar informações atuais na internet."
    ),
    Tool(
        name="Calculadora",
        func=calcular,
        description="Use para cálculos matemáticos simples."
    ),
    Tool(
        name="Tradutor",
        func=traduz,
        description="Use para traduzir textos para português brasileiro, sempre que o input não estiver em português."
    ),
    Tool(
        name="DataHoje",
        func=dataHoje,
        description="sempre que o usuário perguntar algo que envolva a data atual, utilize essa ferramenta para responder com a data de hoje."
    )
]

# =========  MEMÓRIA =========
# Armazena o histórico das conversas (mensagens anteriores)
memory = ConversationBufferMemory(
    memory_key="chat_history",  # nome da variável usada pelo agente
    return_messages=True
)

# =========  AGENTE COM MEMÓRIA =========
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,  # ReAct com memória
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
)

# =========  INTERAÇÃO =========
print("🤖 Agente iniciado! Digite 'sair' para encerrar.\n")

while True:
    pergunta = input("Você: ")
    if pergunta.lower() in ["sair", "exit", "quit"]:
        break

    resposta = agent.run(pergunta)
    print("Agente:", resposta)

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from tcc.config import templates
from fastapi.staticfiles import StaticFiles

from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings

_ = load_dotenv(find_dotenv())

ollama_server_url = "http://localhost:11434"
model_local = ChatOllama(model="llama3.2:3b", base_url=ollama_server_url)

rag_template = """
Você é um especialista em engenharia de prompts.

Sua tarefa é criar um prompt ideal, claro e eficiente, baseado nos seguintes parâmetros fornecidos pelo usuário:

- Persona: {persona}
- Objetivo: {objetivo}
- Roteiro: {roteiro}
- Tom: {tom}
- Panorama: {panorama}
- Idioma: {idioma}
- Modelo Alvo (O prompt deve ser otimizado para esse modelo de IA) : {modelo_alvo}

Utilize o contexto abaixo como referência (base de conhecimento, exemplos anteriores, boas práticas):

Contexto: {context}

Gere apenas o prompt final, sem explicações adicionais.
"""

prompt = ChatPromptTemplate.from_template(rag_template)

loader = CSVLoader(file_path="base.csv")
documents = loader.load()

embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever()


def extrair_query_para_busca(entrada):

    if isinstance(entrada, dict):
        campos_busca = [
            entrada.get("persona", ""),
            entrada.get("objetivo", ""),
            entrada.get("roteiro", ""),
            entrada.get("tom", ""),
            entrada.get("modelo_alvo", "")
        ]

        query_busca = " ".join([campo for campo in campos_busca if campo.strip()])
        return query_busca

    return str(entrada)


def obter_contexto(entrada):

    query = extrair_query_para_busca(entrada)
    documentos = retriever.invoke(query)

    contexto = "\n\n".join([doc.page_content for doc in documentos])
    return contexto


def preparar_dados_prompt(entrada):

    contexto = obter_contexto(entrada)

    return {
        "context": contexto,
        "persona": entrada.get("persona", ""),
        "objetivo": entrada.get("objetivo", ""),
        "roteiro": entrada.get("roteiro", ""),
        "tom": entrada.get("tom", ""),
        "panorama": entrada.get("panorama", ""),
        "idioma": entrada.get("idioma", ""),
        "modelo_alvo": entrada.get("modelo_alvo", "")
    }


chain = (
        RunnableLambda(preparar_dados_prompt)
        | prompt
        | model_local
)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def exibir_formulario(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/gerar-prompt", response_class=HTMLResponse)
async def gerar_prompt(
        request: Request,
        persona: str = Form(...),
        objetivo: str = Form(...),
        roteiro: str = Form(...),
        tom: str = Form(...),
        panorama: str = Form(...),
        idioma: str = Form(...),
        modeloIA: str = Form(...)
):
    try:
        entrada = {
            "persona": persona,
            "objetivo": objetivo,
            "roteiro": roteiro,
            "tom": tom,
            "panorama": panorama,
            "idioma": idioma,
            "modelo_alvo": modeloIA
        }

        print(f"[DEBUG] Entrada recebida: {entrada}")

        resultado = chain.invoke(entrada)

        return templates.TemplateResponse("index.html", {
            "request": request,
            "resultado": resultado.content,
            "entrada": entrada
        })

    except Exception as e:
        print(f"[ERRO] Erro ao gerar prompt: {str(e)}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "erro": f"Erro ao gerar prompt: {str(e)}",
            "entrada": entrada if 'entrada' in locals() else {}
        })
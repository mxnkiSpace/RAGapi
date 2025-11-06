# main.py
import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
from fastapi.middleware.cors import CORSMiddleware

from qdrant_client import QdrantClient
#from qdrant_client.http import models as qm
from qdrant_client import models as qm 
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_classic.schema.document import Document

# Docstore persistente (para padres)
try:
    from langchain.storage import LocalFileStore
except ImportError:
    from langchain_classic.storage.file_system import LocalFileStore

# Opcional: si quieres usar el retriever multi-vector (padres/hijos)
try:
    from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
    HAS_MULTI = True
except Exception:
    HAS_MULTI = False

# ====================== Config / Infra ======================
load_dotenv(override=True)
load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "bases_legales")
QDRANT_VECTOR_NAME = os.getenv("QDRANT_VECTOR_NAME", "text")

SUMMARIES_PATH = Path(os.getenv("SUMMARIES_PATH", "./data/summaries_path/"))
DOCSTORE_PATH = os.getenv("DOCSTORE_PATH", "./data/docstore")

if not OPENAI_API_KEY:
    raise RuntimeError("Falta OPENAI_API_KEY en variables de entorno.")

embeddings = OpenAIEmbeddings(
    model=OPENAI_EMBEDDING_MODEL,
    openai_api_key=OPENAI_API_KEY,
)
llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
)

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
vectorstore = QdrantVectorStore(
    client=qdrant,
    collection_name=QDRANT_COLLECTION,
    embedding=embeddings,
    vector_name=QDRANT_VECTOR_NAME,  # debe existir en la colección
)

docstore = LocalFileStore(DOCSTORE_PATH)  # almacena los "padres" como JSON->bytes

app = FastAPI(title="RAG API", version="1.0.0", docs_url="/", redoc_url=None)


# ====================== Modelos ======================
class AskRequest(BaseModel):
    fondo_id: int
    question: str
    k: int = 4
    doc_hash: Optional[str] = None  # por si quieres acotar a 1 documento


class AskResponse(BaseModel):
    answer: str
    queries: List[str]
    sources: List[Dict[str, Any]]


# ====================== Utilidades ======================

# Se habilita la API para CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def qdrant_search_docs(query: str, k: int, filtro: Dict[str, Any] | None):
    """
    Búsqueda directa en Qdrant usando el cliente oficial.
    Devuelve List[Document] compatibles con tu flujo.
    """
    # 1) Embedding de la consulta
    vec = embeddings.embed_query(query)

    # 2) Search directo (usa vector_name configurado)
    res = qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=(QDRANT_VECTOR_NAME, vec),
        limit=k,
        query_filter=filtro,       # dict estilo {"must": [{"key": "...", "match": {"value": ...}}]}
        with_payload=True,
        with_vectors=False,
    )

    # 3) Mapear resultados a Document
    docs = []
    for r in res:
        payload = r.payload or {}
        # page_content puede estar en 'page_content' (add_documents) o 'text' (add_texts)
        content = (
            payload.get("page_content")
            or payload.get("text")
            or ""  # último recurso
        )
        # metadatos: quita el contenido si está duplicado
        meta = dict(payload)
        for key in ("page_content", "text"):
            if key in meta:
                meta.pop(key)
        docs.append(Document(page_content=content, metadata=meta))
    return docs

def list_fondo_ids_from_summaries(base: Path) -> List[int]:
    if not base.exists():
        return []
    fondos = []
    for p in base.iterdir():
        if p.is_dir():
            try:
                fondos.append(int(p.name))
            except ValueError:
                continue
    return sorted(fondos)


def qdrant_filter_dict(fondo_id: int, doc_hash: str | None):
    must = [{"key": "id_fondo", "match": {"value": int(fondo_id)}}]
    if doc_hash:
        must.append({"key": "doc_hash", "match": {"value": str(doc_hash)}})
    return {"must": must}



from typing import List
import json, re
from langchain_core.messages import SystemMessage, HumanMessage

def _extract_json_array(text: str) -> List[str] | None:
    """
    Intenta extraer y parsear el primer array JSON de la cadena.
    Tolera code fences ```json ... ```, espacios y texto adicional.
    """
    if not text:
        return None

    s = text.strip()

    # Quita fences tipo ```json ... ``` o ```
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.S).strip()

    # Si aún viene con texto alrededor, buscamos el primer bloque [ ... ]
    m = re.search(r"\[\s*(?:\".*?\"(?:\s*,\s*\".*?\")*)\s*\]", s, flags=re.S)
    if m:
        s = m.group(0)

    try:
        arr = json.loads(s)
        if isinstance(arr, list):
            # Normaliza a strings, limpia blancos y deduplica manteniendo orden
            seen, out = set(), []
            for x in arr:
                t = str(x).strip()
                if t and t not in seen:
                    seen.add(t)
                    out.append(t)
            return out
    except Exception:
        return None

def generate_alternative_queries(question: str) -> List[str]:
    """
    Expande la consulta en 3 variantes complementarias.
    Intenta JSON estricto; si falla, usa fallback seguro.
    """
    sys = (
        "Genera exactamente 2 reformulaciones útiles y distintas, técnicas y relevantes"
        "de la pregunta del usuario para mejorar la recuperación de información"
        "en bases legales para fondos concursables en Chile. "
        "Responde SOLO como una lista JSON de strings."
    )
    user = f"Pregunta original: {question}"

    try:
        # Usa mensajes tipados de LangChain (evitamos dicts crudos)
        resp = llm.invoke([
            SystemMessage(content=sys),
            HumanMessage(content=user),
        ])

        text = resp.content if hasattr(resp, "content") else str(resp)
        arr = _extract_json_array(text)

        if arr:
            # Asegura exactamente 3 (si el modelo devuelve más/menos)
            arr = arr[:3] if len(arr) >= 3 else (arr + [question.strip()])[:3]
            return arr
    except Exception:
        # (Opcional) loguea el error para debug
        # print("generate_alternative_queries error:", e)
        pass

    # Fallback simple y consistente
    q = question.strip()
    return [
        q,
        f"{q} (requisitos, plazos, montos, criterios)",
        f"Detalles normativos y excepciones: {q}",
    ]



def fetch_parents(doc_ids: List[str]) -> List[Optional[Dict[str, Any]]]:
    """
    Lee 'padres' desde LocalFileStore. Devuelve dicts (ya decodificados)
    o None si no existe.
    """
    out = []
    for pid in doc_ids:
        try:
            raw = docstore.get(pid)
            if raw is None:
                out.append(None)
            else:
                out.append(json.loads(raw.decode("utf-8")))
        except Exception:
            out.append(None)
    return out

def qdrant_filter(fondo_id: int, doc_hash: str | None):
    must = [qm.FieldCondition(key="id_fondo", match=qm.MatchValue(value=str(fondo_id)))]
    if doc_hash:
        must.append(qm.FieldCondition(key="doc_hash", match=qm.MatchValue(value=str(doc_hash))))
    return qm.Filter(must=must)

def build_answer(context_chunks: List[str], question: str) -> str:
    prompt = (
        "Responde en español con precisión y claridad.\n"
        "Usa SOLO la información del contexto; si algo no está en el contexto, indícalo explícitamente.\n\n"
        f"Contexto:\n{'\n\n'.join(context_chunks)}\n\n"
        f"Pregunta:\n{question}\n\n"
        "Respuesta:"
    )
    return llm.invoke(prompt).content


# ====================== Endpoints ======================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/fondos", response_model=List[int])
def fondos():
    """
    Lista de fondos disponibles para RAG. Por defecto recorre SUMMARIES_PATH
    y devuelve los nombres de subcarpetas numéricas.
    """
    fondos = list_fondo_ids_from_summaries(SUMMARIES_PATH)
    return fondos


@app.post("/chat", response_model=AskResponse)
def chat(req: AskRequest):
    print("Preparandose para responder")
    if not req.question or not req.question.strip():
        raise HTTPException(400, "Pregunta vacía")

    # 1) Expandir consulta
    expansions = generate_alternative_queries(req.question)
    # Incluye la original si no viene
    if req.question not in expansions:
        expansions = [req.question] + expansions
    # limitar a 3
    expansions = expansions[:3]

    # 2) Recuperación por cada expansión (con filtro por fondo/doc_hash)
    print("Preparandose para recuperar... Con filtro:", req.fondo_id)

    filtro = qm.Filter(
        must=[
            qm.FieldCondition(
                key="metadata.id_fondo",
                match=qm.MatchValue(value=int(req.fondo_id))
            )
        ]
    )
    hits: List[Any] = []
    seen_ids = set()
    for q in expansions:
        docs = vectorstore.similarity_search(q, k=req.k, filter=filtro)
        if docs:
            print(f"First result metadata: {docs[0].metadata}")
        # deduplicar por (doc_id, idx) si existe, o por contenido
        for d in docs:
            print(d.metadata.get("id_fondo"), type(d.metadata.get("id_fondo")))
            #if d.metadata.get("")
            key = (d.metadata.get("doc_id"), d.metadata.get("idx"))
            if key not in seen_ids:
                seen_ids.add(key)
                hits.append(d)

    if not hits:
        return AskResponse(
            answer="No se encontró información relevante para esta consulta en el fondo indicado.",
            queries=expansions,
            sources=[]
        )

    # 3) Construir contexto (recorta si quieres)
    context = [h.page_content for h in hits[: max(6, req.k)]]

    # 4) Generar respuesta
    answer = build_answer(context, req.question)

    # 5) Fuentes + opcional 'padres'
    #    Si necesitas “padres”, los cargamos por doc_id desde LocalFileStore
    doc_ids = [h.metadata.get("doc_id") for h in hits]
    parents = fetch_parents([d for d in doc_ids if d])
    parents_iter = iter(parents)

    sources: List[Dict[str, Any]] = []
    for h in hits:
        meta = dict(h.metadata or {})
        parent_obj = next(parents_iter, None) if meta.get("doc_id") else None
        snippet = h.page_content[:300]
        sources.append({
            "kind": meta.get("kind"),
            "doc_id": meta.get("doc_id"),
            "id_fondo": meta.get("id_fondo"),
            "doc_hash": meta.get("doc_hash"),
            "source": meta.get("source"),
            "idx": meta.get("idx"),
            "snippet": snippet,
            "parent_present": parent_obj is not None,
        })

    return AskResponse(answer=answer, queries=expansions, sources=sources)

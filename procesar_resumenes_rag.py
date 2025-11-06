# procesar_resumenes_rag.py
import os
import json
import hashlib
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple
from pydantic import BaseModel

from dotenv import load_dotenv, find_dotenv
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared, operations
from unstructured_client.models.errors import SDKError
from unstructured.staging.base import dict_to_elements

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_classic.schema.document import Document
import uuid

# ============== Utilidades básicas ==============
def _pdf_doc_hash(pdf_path: Path, digest_len: int = 16) -> str:
    h = hashlib.sha256()
    with pdf_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:digest_len]

class Element(BaseModel):
    type: str          # "text" | "table"
    page_content: Any  # str (texto) o HTML (tabla)

# Limpieza de entradas/salidas
_DISCOURAGED_PREFIXES = re.compile(
    r"^\s*(claro|por supuesto|a continuaci[oó]n|aquí(?:\s+te\s+presento| tienes)?|lo siento|no puedo)\b",
    re.IGNORECASE
)

def _sanitize_input(doc: str) -> str:
    if not doc or not doc.strip():
        return ""
    return doc

def _sanitize_output(text: str) -> str:
    if text is None:
        return "SIN_CONTENIDO_UTIL"
    t = text.strip()
    t = _DISCOURAGED_PREFIXES.sub("", t).strip()  # quita muletillas si aparecen
    return t if t else "SIN_CONTENIDO_UTIL"

# Prompt robusto
SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    """Eres un asistente experto en resumir texto y tablas en español para público no experto,
manteniendo los términos legales relevantes cuando aparezcan.

REGLAS:
- No uses prefacios ni muletillas (p. ej., "Claro", "A continuación").
- No pidas más información ni te disculpes.
- No inventes datos ni enlaces. Si algo no está en la entrada, no lo asumas.
- Si el contenido no es una tabla ni texto útil (o está vacío), responde EXACTAMENTE: "SIN_CONTENIDO_UTIL".
- Si es una tabla en HTML (<table>), identifica columnas clave y ofrece 3–6 hallazgos (tendencias, extremos, conteos).
- Si es texto, produce 4–8 viñetas concisas y claras, preservando términos legales si los hay.
- Tono formal y profesional.

ENTRADA:
{doc}

SALIDA (texto plano, sin encabezados, sin preámbulos):"""
)

# ============== Función principal ==============
def procesar_documento(
    pdf_path: str,
    fondo_id: int,
    summaries_path: str,
    *,
    openai_model: str = None,
    unstructured_api_key: str = None,
    unstructured_api_url: str = None,
    force: bool = False
) -> Dict[str, Any]:
    """
    Procesa un PDF, genera (o carga) resúmenes y devuelve artefactos listos para el indexado:
      - child_docs: List[Document] (resúmenes) con metadata {doc_id, kind, id_fondo, doc_hash, source, idx}
      - parent_docs: List[Any] (texto/tabla original) para docstore.mset
      - doc_ids: List[str] alineados con child_docs/parent_docs
      - metadatos: {doc_hash, source_pdf, id_fondo, cache_file}
    """
    # Carga .env si está disponible
    load_dotenv(override=True)
    load_dotenv(find_dotenv())

    OPENAI_MODEL = openai_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    UNSTRUCTURED_API_KEY = unstructured_api_key or os.getenv("UNSTRUCTURED_API_KEY")
    UNSTRUCTURED_API_URL = unstructured_api_url or os.getenv("UNSTRUCTURED_API_URL")

    PDF_PATH = Path(pdf_path)
    if not PDF_PATH.exists() or not PDF_PATH.is_file():
        raise FileNotFoundError(f"No se encontró el PDF: {PDF_PATH}")

    SUMMARIES_PATH = Path(summaries_path)
    SUMMARIES_PATH.mkdir(parents=True, exist_ok=True)

    DOC_HASH = _pdf_doc_hash(PDF_PATH)
    fondo_dir = SUMMARIES_PATH / str(fondo_id)
    fondo_dir.mkdir(parents=True, exist_ok=True)
    out_file = fondo_dir / f"{DOC_HASH}.json"

    # ---------- Unstructured: partición ----------
    client = UnstructuredClient(
        api_key_auth=UNSTRUCTURED_API_KEY,
        server_url=UNSTRUCTURED_API_URL
    )

    with PDF_PATH.open("rb") as f:
        files = shared.Files(content=f.read(), file_name=str(PDF_PATH))

    params = shared.PartitionParameters(
        files=files,
        strategy="hi_res",
        hi_res_model_name="yolox",
        pdf_infer_table_structure=True,
        skip_infer_table_types=[]
    )
    request = operations.PartitionRequest(partition_parameters=params)

    try:
        resp = client.general.partition(request=request)
        elements = dict_to_elements(resp.elements)
    except SDKError as e:
        raise RuntimeError(f"Unstructured SDKError: {e}")

    # ---------- Normalización a Element(text|table) ----------
    categorized_elements: List[Element] = []
    for el in elements:
        cat = getattr(el, "category", None)
        if cat == "Table":
            html = getattr(getattr(el, "metadata", None), "text_as_html", None)
            if html:
                categorized_elements.append(Element(type="table", page_content=html))
        elif cat in {
            "Text", "NarrativeText", "ListItem", "Title", "Address",
            "EmailAddress", "Header", "Footer", "CodeSnippet",
            "UncategorizedText", "FigureCaption", "Formula"
        }:
            txt = getattr(el, "text", None)
            if txt:
                categorized_elements.append(Element(type="text", page_content=txt))
        # Otros tipos se ignoran (imágenes, etc.)

    table_elements = [e for e in categorized_elements if e.type == "table"]
    text_elements  = [e for e in categorized_elements if e.type == "text"]

    # ---------- Cadena de resumen ----------
    summary_chain = (
        {"doc": lambda x: x}
        | SUMMARY_PROMPT
        | ChatOpenAI(model=OPENAI_MODEL, temperature=0.2, max_retries=3)
        | StrOutputParser()
    )

    # ---------- Cache: cargar o crear ----------
    def _load_cache() -> Tuple[List[str], List[str]]:
        with out_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        tables = [x["summary"] for x in data.get("tables", [])]
        texts  = [x["summary"] for x in data.get("texts", [])]
        return tables, texts

    def _save_cache(table_summaries: List[str], text_summaries: List[str]) -> None:
        payload = {
            "id_fondo": fondo_id,
            "document_id": DOC_HASH,
            "source_pdf": str(PDF_PATH),
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tables": [{"index": i, "summary": s} for i, s in enumerate(table_summaries)],
            "texts":  [{"index": i, "summary": s} for i, s in enumerate(text_summaries)],
        }
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    if out_file.exists() and not force:
        table_summaries, text_summaries = _load_cache()
    else:
        tables_content = [_sanitize_input(e.page_content) for e in table_elements]
        text_content   = [_sanitize_input(e.page_content) for e in text_elements]

        table_summaries = []
        if any(bool(c.strip()) for c in tables_content):
            raw = summary_chain.batch(tables_content, {"max_concurrency": 5})
            table_summaries = [_sanitize_output(r) for r in raw]

        text_summaries = []
        if any(bool(c.strip()) for c in text_content):
            raw = summary_chain.batch(text_content, {"max_concurrency": 5})
            text_summaries = [_sanitize_output(r) for r in raw]

        _save_cache(table_summaries, text_summaries)

    # ---------- Preparación para la base vectorial (MultiVectorRetriever) ----------
    child_docs: List[Document] = []
    parent_docs: List[Any] = []
    doc_ids: List[str] = []

    # Tablas
    for i, summary in enumerate(table_summaries):
        did = str(uuid.uuid4())
        child_docs.append(
            Document(
                page_content=summary,
                metadata={
                    "doc_id": did,
                    "kind": "table_summary",
                    "id_fondo": fondo_id,         # <-- para filtro
                    "doc_hash": DOC_HASH,         # <-- para idempotencia/filtro
                    "source": str(PDF_PATH),
                    "idx": i,
                },
            )
        )
        parent_docs.append(table_elements[i].page_content if i < len(table_elements) else "")
        doc_ids.append(did)

    # Textos
    for i, summary in enumerate(text_summaries):
        did = str(uuid.uuid4())
        child_docs.append(
            Document(
                page_content=summary,
                metadata={
                    "doc_id": did,
                    "kind": "text_summary",
                    "id_fondo": fondo_id,         # <-- para filtro
                    "doc_hash": DOC_HASH,         # <-- para idempotencia/filtro
                    "source": str(PDF_PATH),
                    "idx": i,
                },
            )
        )
        parent_docs.append(text_elements[i].page_content if i < len(text_elements) else "")
        doc_ids.append(did)

    return {
        "child_docs": child_docs,       # para vectorstore.add_documents(...)
        "parent_docs": parent_docs,     # para docstore.mset(list(zip(doc_ids, parent_docs)))
        "doc_ids": doc_ids,             # alineados con child_docs y parent_docs
        "doc_hash": DOC_HASH,
        "source_pdf": str(PDF_PATH),
        "id_fondo": fondo_id,
        "cache_file": str(out_file),
        "counts": {
            "tables": len(table_summaries),
            "texts": len(text_summaries),
            "child_docs": len(child_docs)
        }
    }

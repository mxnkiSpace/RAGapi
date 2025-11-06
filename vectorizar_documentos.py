# main_index_qdrant.py
import os
import json
from dotenv import load_dotenv, find_dotenv
load_dotenv(override=True)
load_dotenv(find_dotenv())

from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_qdrant import QdrantVectorStore  # ✅ vectorstore actual
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from qdrant_client.http import models as qm
from qdrant_client import QdrantClient
#rom langchain_core.stores import InMemoryStore  # o LocalFileStore si quieres persistencia
# from langchain.storage import LocalFileStore
try:
    from langchain.storage import LocalFileStore
except ImportError:
    # fallback para entornos "classic"
    from langchain_classic.storage.file_system import LocalFileStore
from procesar_resumenes_rag import procesar_documento

# ---------- Embeddings ----------
embeddings = OpenAIEmbeddings(
    model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),   # ✅ parámetro correcto
)
dim = len(embeddings.embed_query("ping"))  # e.g., 1536

# ---------- Qdrant client ----------
COLLECTION = "bases_legales"
VECTOR_NAME = "text"  # ✅ vamos a usar nombre explícito

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    api_key=os.getenv("QDRANT_API_KEY")  # None si sin auth
)

# qdrant.delete(
#     collection_name=COLLECTION,
#     points_selector=qm.FilterSelector(
#     filter=qm.Filter(
#         must=[] # An empty 'must' list in a filter matches all points
#         )
#      ),
#      wait=True # Set to True to wait for the operation to complete
# )


# Crear colección si no existe, con named vector "text"
try:
    qdrant.get_collection(COLLECTION)
except Exception:
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config={VECTOR_NAME: qm.VectorParams(size=dim, distance=qm.Distance.COSINE)},
    )

# vectorizar_documentos.py (Línea 65)
qdrant.create_payload_index(
    collection_name=COLLECTION,
    field_name="id_fondo",
    field_schema=qm.PayloadSchemaType.INTEGER
)
# ---------- Procesar documento ----------
##########################################
##########################################
##########################################
##########################################
result = procesar_documento(
    pdf_path="./data/documents/beneficios_complementarios_2026.pdf",
    fondo_id=3,
    summaries_path="./data/summaries_path/",
    openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    force=False,
)

##########################################
##########################################
##########################################
##########################################
child_docs  = result["child_docs"]
parent_docs = result["parent_docs"]
doc_ids     = result["doc_ids"]

# ---------- VectorStore (Qdrant) ----------
vectorstore = QdrantVectorStore(
    client=qdrant,
    collection_name=COLLECTION,   # "bases_legales"
    embedding=embeddings,         # <-- OJO: 'embedding', no 'embeddings'
    vector_name=VECTOR_NAME,      # debe coincidir con el nombre del vector de la colección
)

# ---------- Retriever Multi-Vector ----------
store = LocalFileStore("./data/docstore")
id_key = "doc_id"

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

# ---------- Indexación ----------

qdrant.delete(
    collection_name=COLLECTION,
    points_selector=qm.FilterSelector(
        filter=qm.Filter(
            must=[qm.FieldCondition(key="doc_hash", match=qm.MatchValue(value=result["doc_hash"]))]
        )
    ),
)


vectorstore.add_documents(child_docs)
pairs = []
for did, doc in zip(doc_ids, parent_docs):
    # Si es un objeto Document de LangChain
    if hasattr(doc, "dict"):
        data = doc.dict()
    # Si es string o cualquier otro tipo
    elif isinstance(doc, str):
        data = {"content": doc}
    else:
        data = {"content": str(doc)}

    # Serializamos a JSON y convertimos a bytes
    pairs.append((did, json.dumps(data, ensure_ascii=False).encode("utf-8")))

retriever.docstore.mset(pairs)

print("Indexación completada en Qdrant.")
print(f"Documento: {result['source_pdf']} | hash: {result['doc_hash']}")
print(f"Resúmenes: {result['counts']}")

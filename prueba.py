from qdrant_client.http import models as qm

query = "¿Cuál es el objetivo?"
filtro = qm.Filter(
    must=[qm.FieldCondition(key="id_fondo", match=qm.MatchValue(value=3))]
)

docs = vectorstore.similarity_search(query, k=5, filter=filtro)
print(len(docs))
for d in docs:
    print(d.page_content[:200], d.metadata)

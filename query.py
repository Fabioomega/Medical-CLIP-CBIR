from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, ScoredPoint, VectorParams, Distance
from embgen import ImageEmbeddingGenerator
from typing import List


def save(
    paths: List[str],
    client: QdrantClient,
    emb_generator: ImageEmbeddingGenerator,
    base_idx: int = -1,
):
    embeddings = emb_generator(paths)

    client.upsert(
        collection_name="3D-tomography",
        points=[
            PointStruct(
                id=idx if base_idx == -1 else idx + base_idx,
                vector=vector.tolist(),
                payload={"path": paths[idx]},
            )
            for idx, vector in enumerate(embeddings)
        ],
    )


def query(
    path: str,
    client: QdrantClient,
    emb_generator: ImageEmbeddingGenerator,
    limit: int = 5,
) -> List[ScoredPoint]:

    emb = emb_generator([path])[0]

    hits = client.search(
        collection_name="3D-tomography",
        query_vector=emb.tolist(),
        limit=limit,  # Return 5 closest points
    )

    return hits


client = QdrantClient(path="images")
generator = ImageEmbeddingGenerator()
generator.load("CT-CLIP_v2.pt")

for q in query("valid_1_a_2.nii.gz", client, generator, limit=6)[1:]:
    path = q.payload["path"]
    print(f"Score: {q.score:.3f}. Path: {path}")

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, ScoredPoint, VectorParams, Distance
from embgen import ImageEmbeddingGenerator
from typing import List
from data import CTReportDatasetinfer
from torch.utils.data import DataLoader
import tqdm


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
    path: str, client: QdrantClient, emb_generator: ImageEmbeddingGenerator
) -> List[ScoredPoint]:

    emb = emb_generator([path])[0]

    hits = client.search(
        collection_name="3D-tomography",
        query_vector=emb.tolist(),
        limit=5,  # Return 5 closest points
    )

    return hits


if __name__ == "__main__":
    client = QdrantClient(path="images")
    if not client.collection_exists("3D-tomography"):
        client.create_collection(
            "3D-tomography",
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )
    generator = ImageEmbeddingGenerator()
    generator.load("CT-CLIP_v2.pt")

    # Prepare the evaluation dataset
    ds = CTReportDatasetinfer(
        data_folder=r"D:\GigaModels\CT-CLIP\data_volumes\dataset\valid",
        csv_file=r"dataset_radiology_text_reports_validation_reports.csv",
    )
    dl = DataLoader(ds, num_workers=2, batch_size=1, shuffle=False)

    idx = 0

    for batch in tqdm.tqdm(dl):
        _, nii_file = batch
        save(nii_file, client, generator, idx)
        idx += len(nii_file)

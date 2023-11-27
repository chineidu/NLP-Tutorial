from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Filter
from sentence_transformers import SentenceTransformer


class NeuralSearcher:
    def __init__(self, collection_name: str) -> None:
        self.collection_name: str = collection_name
        # Initialize encoder model
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        # initialize Qdrant client
        self.qdrant_client: QdrantClient = QdrantClient("http://localhost:6333")

    def search(
        self,
        text: str,
        city_of_interest: str,
    ) -> list[dict[str, Any]]:
        # Convert text query into vector
        vector = self.encoder.encode(text).tolist()
        if city_of_interest is not None:
            city_of_interest = city_of_interest.title()

        # Define a filter for cities
        city_filter = Filter(
            **{
                "must": [
                    {
                        "key": "city",  # Store city information in a field of the same name
                        "match": {  # This condition checks if payload field has the requested value
                            "value": city_of_interest
                        },
                    }
                ]
            }
        )
        query_filter = None if city_of_interest == "" else city_filter
        # Use `vector` for search for closest vectors in the collection
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=query_filter,
            limit=10,
        )
        # `search_result` contains found vector ids with
        # similarity scores along with the stored payload
        payloads = [{**hit.payload, **{"score": hit.score}} for hit in search_result]
        return payloads


app = FastAPI()


class InputSchema(BaseModel):
    query: str
    city_of_interest: str

    class Config:
        """Sample Payload"""

        schema_extra = {"example": {"query": "Google", "city_of_interest": "New York"}}


# Create a neural searcher instance
neural_searcher = NeuralSearcher(collection_name="startups")


@app.post("/api/v1/search")
def search_startup(data: InputSchema) -> dict[str, list[dict[str, Any]]]:
    query: str = data.query
    city_of_interest: str = data.city_of_interest

    return {
        "result": neural_searcher.search(
            text=query,
            city_of_interest=city_of_interest,
        )
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("neural_searcher:app", host="0.0.0.0", port=8000, reload=True)

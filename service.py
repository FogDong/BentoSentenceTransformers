from __future__ import annotations

import typing as t

import numpy as np
import bentoml


SAMPLE_SENTENCE = "The sun dips below the horizon, painting the sky orange."

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
BENTO_MODEL_TAG = MODEL_ID.lower().replace("/", "--")


@bentoml.service(
    traffic={"timeout": 60},
    resources={"memory": "2Gi"},
)
class SentenceTransformers:

    model = bentoml.models.get(BENTO_MODEL_TAG)

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.st = SentenceTransformer(self.model.path, device="cpu")
        print(f"Model '{MODEL_ID}' loaded on device: '{self.device}'.")

    @bentoml.api()
    def encode(
        self,
        data: t.List = [[0, SAMPLE_SENTENCE]],
    ):
        print("data:", data)
        input_text = [item[1] for item in data]
        print("input_text:", input_text)

        result = self.st.encode(input_text)
        data = {"data": [[index, value.tolist()] for index, value in enumerate(result)]}
        return data

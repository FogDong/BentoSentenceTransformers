import bentoml

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
BENTO_MODEL_TAG = MODEL_ID.lower().replace("/", "--")


def import_model(model_id, bento_model_tag):

    import torch
    from sentence_transformers import SentenceTransformer, models

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_ID, device=device)

    with bentoml.models.create(bento_model_tag) as bento_model_ref:
        model.save(bento_model_ref.path)


if __name__ == "__main__":
    import_model(MODEL_ID, BENTO_MODEL_TAG)

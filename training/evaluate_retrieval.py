"""
Evaluate retrieval quality: fine-tuned domain embedding model vs. the
generic baseline it was fine-tuned from.

Metric: Recall@3 - for each test query, does the correct passage appear
in the top-3 retrieved passages? (Tighter than top-5 so a real gap
between models has room to show up.)

IMPORTANT: TEST_QUERIES below are deliberately phrased differently from
anything in training_pairs.jsonl - they are a genuine held-out set, not
data the model has seen. Reusing training phrasings here would give an
inflated, dishonest score.
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

BASE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FINE_TUNED_PATH = Path(__file__).parent.parent / "models" / "fine_tuned_embedder"
DOCS_PATH = Path(__file__).parent.parent / "data" / "raw_documents" / "hvac_faq.txt"

# Held-out test set: (query, index_of_correct_passage_in_corpus)
# Index corresponds to the passage's position (0-based) among the 35
# Q&A blocks in hvac_faq.txt, in file order.
TEST_QUERIES = [
    ("no cool air at all coming out of the vents", 0),
    ("do I need to swap the filter often", 1),
    ("furnace won't stop clicking on and off", 2),
    ("ballpark cost for a full replacement", 3),
    ("what does a higher efficiency rating mean for my bill", 4),
    ("nothing shows up on the thermostat screen", 5),
    ("can someone come same night if the heat is out", 6),
    ("how many hours to put in a new AC unit", 7),
    ("do you have an annual service package", 8),
    ("frost building up on the outdoor unit", 9),
    ("heater is totally unresponsive, nothing happens", 10),
    ("hearing a clanking sound from the system", 11),
    ("steps to reset the breaker myself", 12),
    ("pros and cons of heat pump vs gas furnace", 13),
    ("putting a unit in a room with no ducts", 14),
    ("ways to cut down dust and allergens inside", 15),
    ("home feels clammy even though AC is on", 16),
    ("static shocks and dry air all winter", 17),
    ("compatibility of ecobee with my current setup", 18),
    ("monthly payment options for a new unit", 19),
    ("what's protected if something breaks after install", 20),
    ("how many years between duct cleanings", 21),
    ("necessity of a CO alarm by heating equipment", 22),
    ("setting up a weekday vs weekend schedule", 23),
    ("cooling system won't stay running for long", 24),
    ("calculating the right tonnage for my square footage", 25),
    ("incentives for upgrading to efficient equipment", 26),
    ("does adding UV lighting actually help air quality", 27),
    ("separate temperature control for upstairs and downstairs", 28),
    ("heat pump pushing out cold air on heat mode", 29),
    ("value of a checkup even when nothing seems wrong", 30),
    ("temperature not matching what's set on the thermostat", 31),
    ("puddle forming near the indoor air handler", 32),
    ("faint gas odor coming from the furnace closet", 33),
    ("burning smell right when the AC starts up", 34),
]


def load_corpus(path: Path) -> list[str]:
    text = path.read_text()
    blocks = [b.strip() for b in text.split("Q:") if b.strip()]
    return ["Q:" + b for b in blocks]


def recall_at_k(model: SentenceTransformer, corpus: list[str], k: int = 3) -> float:
    corpus_embeddings = model.encode(corpus, convert_to_numpy=True)
    hits = 0
    for query, correct_idx in TEST_QUERIES:
        query_embedding = model.encode(query, convert_to_numpy=True)
        sims = corpus_embeddings @ query_embedding / (
            np.linalg.norm(corpus_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        top_k = np.argsort(-sims)[:k]
        if correct_idx in top_k:
            hits += 1
    return hits / len(TEST_QUERIES)


def main():
    corpus = load_corpus(DOCS_PATH)
    print(f"Loaded {len(corpus)} passages from corpus")
    print(f"Evaluating on {len(TEST_QUERIES)} held-out test queries\n")

    print(f"Evaluating baseline model: {BASE_MODEL_NAME}")
    baseline_model = SentenceTransformer(BASE_MODEL_NAME)
    baseline_recall = recall_at_k(baseline_model, corpus)
    print(f"Baseline Recall@3: {baseline_recall:.2f}")

    if not FINE_TUNED_PATH.exists():
        print(f"\nNo fine-tuned model found at {FINE_TUNED_PATH}.")
        print("Run training/fine_tune_embeddings.py first.")
        return

    print(f"\nEvaluating fine-tuned model: {FINE_TUNED_PATH}")
    fine_tuned_model = SentenceTransformer(str(FINE_TUNED_PATH))
    fine_tuned_recall = recall_at_k(fine_tuned_model, corpus)
    print(f"Fine-tuned Recall@3: {fine_tuned_recall:.2f}")

    delta = (fine_tuned_recall - baseline_recall) * 100
    print(f"\nImprovement: {delta:+.1f} percentage points")
    print("\n>>> Use these real numbers in your README and CV, not placeholders. <<<")


if __name__ == "__main__":
    main()

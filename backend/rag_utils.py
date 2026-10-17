
from langchain_openai import ChatOpenAI
import os
import re
import math
from collections import Counter


# Common words that do not identify the policy topic
STOP_WORDS = set("""
a an the is are was were i my me we our you your it its
to of at in on for from with and or but what when where
how do does did can could should would have has been
since supposed next door
""".split())


def search_words(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [word for word in words if word not in STOP_WORDS]


# Retrieve relevant chunks using similarity and keyword matching
def retrieve(query: str, k: int = 4, vectorstore=None) -> list[str]:

    if vectorstore is None:
        raise ValueError("A vector store is required.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    requested_codes = set(
        re.findall(r"\bASH-[A-Z]{3}\b", query.upper())
    )

    # Get all stored chunks in similarity order
    ranked_results = vectorstore.similarity_search(
        query,
        k=vectorstore.index.ntotal
    )

    if not ranked_results:
        return []

    similarity_ranks = {
        id(chunk): rank
        for rank, chunk in enumerate(ranked_results, start=1)
    }

    # Build keyword counts from the saved chunks
    chunk_words = [
        Counter(search_words(chunk.page_content))
        for chunk in ranked_results
    ]

    document_counts = Counter()

    for words in chunk_words:
        document_counts.update(words.keys())

    query_words = set(search_words(query))

    # Connect everyday language with policy wording
    noise_words = {
        "loud", "noisy", "noise", "quiet",
        "disturbance", "disturbances"
    }

    if query_words & noise_words:
        query_words.update(noise_words)

    keyword_scores = {}

    for chunk, words in zip(ranked_results, chunk_words):
        score = 0.0

        for word in query_words:
            if word in words:
                score += math.log(
                    1 + len(ranked_results) / document_counts[word]
                )

        text = chunk.page_content
        keyword_scores[text] = max(
            score,
            keyword_scores.get(text, 0.0)
        )

    highest_keyword_score = max(
        keyword_scores.values(),
        default=0.0
    )

    # Combine keyword matching with similarity ranking
    def combined_score(chunk):
        similarity_score = (
            60 / (60 + similarity_ranks[id(chunk)])
        )

        if highest_keyword_score == 0:
            return similarity_score

        keyword_score = (
            keyword_scores.get(chunk.page_content, 0.0)
            / highest_keyword_score
        )

        return (
            0.75 * keyword_score
            + 0.25 * similarity_score
        )

    ranked_results.sort(
        key=combined_score,
        reverse=True
    )

    selected = []
    seen_text = set()
    included_properties = set()

    def add_chunk(chunk):
        normalized_text = " ".join(
            chunk.page_content.lower().split()
        )

        if normalized_text not in seen_text:
            selected.append(chunk)
            seen_text.add(normalized_text)

    # First include a matching row for each requested hotel
    if requested_codes:
        for chunk in ranked_results:
            if chunk.metadata.get("content_type") != "table_row":
                continue

            row_codes = set(
                re.findall(
                    r"\bASH-[A-Z]{3}\b",
                    chunk.page_content.upper()
                )
            )

            matching_codes = requested_codes & row_codes

            if matching_codes - included_properties:
                add_chunk(chunk)
                included_properties.update(matching_codes)

            if len(selected) >= k:
                break

    # Fill remaining places with related policy information
    for chunk in ranked_results:
        if len(selected) >= k:
            break

        if (
            requested_codes
            and chunk.metadata.get("content_type") == "table_row"
        ):
            row_codes = set(
                re.findall(
                    r"\bASH-[A-Z]{3}\b",
                    chunk.page_content.upper()
                )
            )

            if row_codes:
                continue

        add_chunk(chunk)

    return [
        f"Content:\n{chunk.page_content}\n\nMetadata:\n{chunk.metadata}"
        for chunk in selected[:k]
    ]


# Generate an answer using the retrieved information
def generate(query, retrieved_chunks, model):

    prompt = f"""
    You are an AI assistant helping AuroraStay hotel staff
    answer questions about hotel policies.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    1. Use only information explicitly supported by the retrieved policy information.
    2. Do not use outside knowledge or make assumptions.
    3. Include the relevant information needed to answer the question completely.
    4. Preserve important conditions, restrictions, exceptions, and approval requirements.
    5. Use the rules for the specific property and rate plan requested.
    6. Do not apply another property's fees or policies to the requested property.
    7. Answer the user's question directly without unrelated information.
    8. Keep the answer concise, clear, and easy to understand.
    9. Do not invent fees, amenities, refund eligibility, deadlines, or guarantees.
    10. If information is insufficient, say:
        "The available policy documents do not contain enough information to answer this question."
        If the retrieved information conflicts, clearly mention the conflict.
    11. Identify the supporting source document or documents using the provided metadata.

    Answer:
    """

    return model.invoke(prompt)


# Combine retrieval and answer generation
def rag(
    query: str,
    k: int = 4,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    api_base = (os.getenv("OPENAI_API_BASE") or "").strip()

    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing.")

    if not api_base:
        raise ValueError("OPENAI_API_BASE is missing.")

    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=api_base
    )

    retrieved_chunks = retrieve(
        query=query,
        k=k,
        vectorstore=vectorstore
    )

    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks

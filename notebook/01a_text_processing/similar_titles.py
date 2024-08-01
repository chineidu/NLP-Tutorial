import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, Dict, List

import polars as pl
from datasketch import LeanMinHash, MinHash, MinHashLSH

# Polars settings
pl.Config.set_fmt_str_lengths(1_000)
pl.Config.set_tbl_cols(n=1_000)
pl.Config.set_tbl_rows(500)


def create_minhash(text: str, num_perm: int = 256) -> LeanMinHash:
    """
    Create a MinHash object from a string using n-grams.

    Parameters
    ----------
    text : str
        The input text to create MinHash from.
    num_perm : int, optional
        Number of permutations for MinHash, by default 256.

    Returns
    -------
    LeanMinHash
        A LeanMinHash object representing the input text.
    """
    minhash = MinHash(num_perm=num_perm)
    # Create 3-grams
    n = 3
    for i in range(len(text) - n + 1):
        ngram = text[i : i + n]
        minhash.update(ngram.encode("utf-8"))
    return LeanMinHash(minhash)


def process_chunk(chunk: List[str], chunk_id: int, num_perm: int = 256) -> List[Dict[str, Any]]:
    """
    Process a chunk of titles and return their MinHashes.

    Parameters
    ----------
    chunk : List[str]
        A list of titles to process.
    chunk_id : int
        The ID of the current chunk.
    num_perm : int, optional
        Number of permutations for MinHash, by default 256.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries containing the ID, title, and MinHash for each title.
    """
    return [
        {
            "id": f"{chunk_id}_{i}",
            "title": title,
            "minhash": create_minhash(title, num_perm),
        }
        for i, title in enumerate(chunk)
    ]


def find_similar_pairs(
    lsh: MinHashLSH, minhashes: List[Dict[str, Any]], threshold: float
) -> List[Dict[str, Any]]:
    """
    Find similar pairs using LSH.

    Parameters
    ----------
    lsh : MinHashLSH
        The LSH index to query.
    minhashes : List[Dict[str, Any]]
        A list of dictionaries containing MinHash information.
    threshold : float
        The similarity threshold for including pairs.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries containing similar title pairs and their similarity scores.
    """
    similar_pairs = []
    for i, item in enumerate(minhashes):
        if i % 1000 == 0:  # Progress update
            print(f"Processed {i}/{len(minhashes)} items")  # noqa
        candidates = lsh.query(item["minhash"])
        for candidate_id in candidates:
            if candidate_id > item["id"]:  # Avoid duplicate pairs
                candidate = next(m for m in minhashes if m["id"] == candidate_id)
                similarity = item["minhash"].jaccard(candidate["minhash"])
                if similarity >= threshold:
                    similar_pairs.append(
                        {
                            "title_1": item["title"],
                            "title_2": candidate["title"],
                            "similarity": round(similarity, 4),
                        }
                    )
    return similar_pairs


def compute_similarity_matrix_fast(
    unique_titles: List[str],
    threshold: float = 0.6,
    chunk_size: int = 10_000,
    num_perm: int = 256,
) -> pl.DataFrame:
    """
    Compute approximate similarity matrix for unique titles using MinHash and LSH.

    Parameters
    ----------
    unique_titles : List[str]
        List of unique titles to compare.
    threshold : float, optional
        Similarity threshold for including pairs, by default 0.6.
    chunk_size : int, optional
        Size of chunks for parallel processing, by default 10000.
    num_perm : int, optional
        Number of permutations for MinHash, by default 256.

    Returns
    -------
    pl.DataFrame
        DataFrame containing similarity scores for title pairs above the threshold.
        Shape: (n_similar_pairs, 3)
        Columns: ['title_1', 'title_2', 'similarity']
    """
    start_time = time.time()
    print(f"Starting similarity computation for {len(unique_titles)} unique titles...")  # noqa

    schema: Dict[str, Any] = {"title_1": str, "title_2": str, "similarity": pl.Float32}

    # Split titles into chunks for parallel processing
    chunks = [unique_titles[i : i + chunk_size] for i in range(0, len(unique_titles), chunk_size)]
    print(f"Split data into {len(chunks)} chunk(s) of size {chunk_size:,}")  # noqa

    # Process chunks in parallel
    print("Creating MinHashes...")  # noqa
    with ProcessPoolExecutor() as executor:
        minhash_results = list(
            executor.map(partial(process_chunk, num_perm=num_perm), chunks, range(len(chunks)))
        )

    # Flatten results
    all_minhashes = [item for sublist in minhash_results for item in sublist]
    print(f"Created {len(all_minhashes)} MinHashes")  # noqa

    # Create LSH index
    print("Building LSH index...")  # noqa
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for item in all_minhashes:
        lsh.insert(item["id"], item["minhash"])

    # Find similar pairs
    print("Finding similar pairs...")  # noqa
    similar_pairs = find_similar_pairs(lsh, all_minhashes, threshold)

    print(f"Found {len(similar_pairs)} similar pairs")  # noqa

    # Create DataFrame
    print("Creating final DataFrame...")  # noqa
    similarity_df: pl.DataFrame = pl.DataFrame(similar_pairs, schema=schema).sort(
        by="title_1", descending=False
    )

    end_time = time.time()
    print(f"\n[INFO]: Computation completed in {end_time - start_time:.2f} seconds")  # noqa

    return similarity_df


if __name__ == "__main__":
    # Load data
    data_path: str = "music_data_vsm.parquet"
    df: pl.DataFrame = pl.scan_parquet(data_path).collect()
    df_music_cleaned = df.with_columns(title=pl.col("title").str.to_lowercase())
    print("[INFO]: Fetching unique titles")  # noqa
    uniq_titles: list[str] = df["title"].unique().to_list()
    print("[INFO]: Computing similarity matrix")  # noqa
    chunk_size: int = int(df.shape[0] / 3)
    similarity_df: pl.DataFrame = compute_similarity_matrix_fast(
        unique_titles=uniq_titles, threshold=0.9, chunk_size=chunk_size, num_perm=256
    )
    similarity_df.write_csv("similar_titles.csv")
    print("[INFO]: Saved to similar_titles.csv")  # noqa

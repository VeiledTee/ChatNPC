import numpy as np
from datetime import datetime
import pinecone

from global_functions import embed

date_format = "%Y-%m-%d %H:%M:%S"


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculates the cosine similarity between two input vectors
    :param a: 1-D array object
    :param b: 1-D array object
    :return: scalar value representing the cosine similarity between the input vectors
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def exponential_decay(current_time: datetime, earlier_time: datetime, decay_rate=0.01) -> float:
    time_difference = (current_time - earlier_time).total_seconds()
    score = 1 / (1 + decay_rate * time_difference)
    return score


def recency_score(record_recent_access: str, cur_time: datetime) -> float:
    """
    Calculates the recency value a record has to the current query.
    Leverages exponential decay from the time the current query was posed to assign a score.
    :param record_recent_access: When the current record was last accessed
    :param cur_time: The time when the user's query was posed to the character
    :return: The recency score of the presented record
    """
    return exponential_decay(cur_time, datetime.strptime(record_recent_access, date_format))


def importance_score(record: dict) -> float:
    """
    Retrieves the importance (poignancy) of the presented record
    :param record: The current record to determine an importance score for
    :return: The poignancy of the presented record
    """
    return record['metadata']['poignancy']


def relevance_score(record_embedding: list, query_embedding: list) -> float:
    """
    Calculates the relatedness of the presented record and the user's query
    :param record_embedding: The embedding representation of the text associated with the current record
    :param query_embedding: The embedding representation of the user's query
    :return: The relevance score of the presented record
    """
    return cos_sim(np.array(record_embedding), np.array(query_embedding))


def retrieval(namespace: str, query_embedding: list[float], n: int, index_name: str = 'thesis-index') -> list[str]:
    """
    Ranks character memories by a retrieval score.
    Retrieval score calculated by multiplying their importance, recency, and relevance scores together.
    Selects the top n of these records to be used as context when replying to the user's query.
    :return: Text from the n memories
    """
    # get index to query for records
    index: pinecone.Index = pinecone.Index(index_name)
    # count number of vectors in the namespace
    total_vectors: int = index.describe_index_stats()["namespaces"][namespace]["vector_count"]
    # query Pinecone and get all records in namespace
    responses = index.query(
        query_embedding,
        top_k=total_vectors + 1,  # +1 to get all vectors in a namespace
        include_metadata=True,
        namespace=namespace,
        filter={
            "$or": [
                {"type": {"$eq": "background"}},
                {"type": {"$eq": "response"}},
            ]
        },
    )
    # find current access time
    cur_time = datetime.now()
    # calculate retrieval score and keep track of record IDs
    # score_id_pairs format: [SCORE, RECORD]
    score_id_pairs: list = [
            (recency_score(x['last_accessed'], cur_time) * importance_score(x) * x['score'], x)
            for x in responses["matches"]
    ]
    # sort records by retrieval score
    sorted_score_id_pairs = sorted(score_id_pairs, key=lambda pair: pair[0], reverse=True)
    # select top n memories
    top_records = [x['metadata']['text'] for x in sorted_score_id_pairs[:n]]

    # update records with new access times
    for _, record in enumerate(top_records):
        index.update(id=record['id'],
                     set_metadata={'last_accessed': str(cur_time)},
                     namespace=namespace
                     )

    return top_records

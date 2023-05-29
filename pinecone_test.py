import pinecone
import numpy as np

with open('keys.txt', 'r') as key_file:
    pinecone.init(api_key=key_file.readlines()[1], environment=key_file.readlines()[2])

metadata_config = {"indexed": ["color"]}

if len(pinecone.list_indexes()) == 0:  # if no indices
    # create index
    pinecone.create_index(
        "thesis-index", dimension=1024, metadata_config=metadata_config
    )
else:  # if indices already exist
    print(f"Index description: {pinecone.describe_index('thesis-index')}")
# list of all indices
active_indexes = pinecone.list_indexes()
print(f"Indices: {active_indexes}")
# retrieve index
index = pinecone.Index(active_indexes[0])
# adding vectors
index.upsert(
    vectors=[
        (
            "vec1",  # Vector ID
            list(np.random.random(1024)),  # Dense vector values
            {"genre": "drama"},  # Vector metadata
        ),
        ("vec2", list(np.random.random(1024)), {"genre": "action"}),
    ],
    namespace="test-namespace",
)
index.upsert(
    vectors=[
        (
            "vec3",  # Vector ID
            list(np.random.random(1024)),  # Dense vector values
            {"genre": "horror"},  # Vector metadata
        ),
        ("vec4", list(np.random.random(1024)), {"genre": "comedy"}),
    ],
    namespace="test-namespace-2",
)
# # delete index
pinecone.delete_index("thesis-index")
# list of all indices
active_indexes = pinecone.list_indexes()
print(f"Indices: {active_indexes}")

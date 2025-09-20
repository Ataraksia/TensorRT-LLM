import json
from pymilvus import connections, Collection

CLUSTER_ENDPOINT = "https://in05-e2d375763d4e834.serverless.gcp-us-west1.cloud.zilliz.com"
TOKEN = "2724fccd4dfaeb4f8b696d3dd05080c9d1a22a3f1191f0e8e08e3d6a1fe18fefe0ff057857ffad4d08477164688b1908ad4402b5"

# Connect with increased gRPC max receive message length (in bytes)
connections.connect(
    uri=CLUSTER_ENDPOINT,
    token=TOKEN,
    grpc_scoped_cfg={
        "grpc.max_receive_message_length": 50
        * 1024
        * 1024  # 50MB; adjust as needed (e.g., 100MB for very large data)
    },
)

collection = Collection("hybrid_code_chunks_6d96a0b9")
total_entities = collection.num_entities
print(f"Total entities: {total_entities}")
# 6. Query with iterator

# Initiate an empty JSON file
with open("results.json", "w") as fp:
    fp.write(json.dumps([]))

iterator = collection.query_iterator(output_fields=["*"], limit=total_entities, batch_size=100)

while True:
    result = iterator.next()
    if not result:
        iterator.close()
        break

    # Read existing records and append the returns
    with open("results.json", "r") as fp:
        results = json.loads(fp.read())
        results += result

    # Save the result set
    with open("results.json", "w") as fp:
        fp.write(json.dumps(results))

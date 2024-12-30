import json
import chromadb
import os
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from vertexai.generative_models import GenerativeModel
from vertexai.vision_models import MultiModalEmbeddingModel
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader


embedding_function = OpenCLIPEmbeddingFunction()
data_loader = ImageLoader()

DEFAULT_DIM = 128
DEFAULT_EMBEDDING = [0.0] * DEFAULT_DIM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSIST_DIRECTORY = os.path.join(ROOT, "product-recommender", "chroma")
COLLECTION_NAME = "db"
N_RECOMMENDATIONS = 3
INPUT_FILE = os.path.join(ROOT, "product-recommender", "db.json")
llm = GenerativeModel("gemini-2.0-flash-exp")


# CREATE TEXT AND IMAGE COLLECTIONS
def create_collection(input_file, output_file, collection_name):
    with open(input_file, "r") as file:
        data = json.load(file)

    client = chromadb.PersistentClient(
        path=output_file,
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )

    img_collection = client.get_or_create_collection(
        name=f"{collection_name}_img_collection",
        embedding_function=embedding_function,
        data_loader=data_loader,
    )
    txt_collection = client.get_or_create_collection(
        name=f"{collection_name}_txt_collection",
        embedding_function=embedding_function,
        data_loader=data_loader,
    )

    img_ids = [f"{item['id']} - {collection_name}_img" for item in data]
    txt_ids = [f"{item['id']} - {collection_name}_txt" for item in data]
    metadata = [item["metadata"] for item in data]
    for meta in metadata:
        for key, value in meta.items():
            if value is None:
                meta[key] = ""
    docs = [item["product_content"] for item in data]
    text_embed = [item["text_embedding"] for item in data]
    image_embed = [item["image_embedding"] for item in data]

    os.makedirs(output_file, exist_ok=True)

    txt_collection.upsert(
        embeddings=text_embed,
        documents=docs,
        metadatas=metadata,
        ids=txt_ids,
    )
    img_collection.upsert(
        embeddings=image_embed,
        documents=docs,
        metadatas=metadata,
        ids=img_ids,
    )


def query_description(query):
    """
    Determines if the query is textually or visually descriptive.
    Returns True if the query is text-based, False if it's image-based.
    """
    prompt = (
        f"Is the following query better suited for text or image-based comparison?\nQuery: {query}\n"
        "Answer 'Text' if text-based or 'Image' if image-based."
    )

    response = llm.generate_content(prompt).text.strip().lower()
    return response == "text"


# FETCH SIMILARITY RESULTS (task 1)
def similarity_results(query, collection_name, output_file, n_res):
    client = chromadb.PersistentClient(
        path=output_file,
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )

    img_collection = client.get_collection(name=collection_name + "_img_collection")
    txt_collection = client.get_collection(name=collection_name + "_txt_collection")

    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    query_embedding = model.get_embeddings(
        contextual_text=query, dimension=DEFAULT_DIM
    ).text_embedding

    txt_results = txt_collection.query(query_embeddings=[query_embedding], n_results=1)
    img_results = img_collection.query(query_embeddings=[query_embedding], n_results=1)

    flag = query_description(query)
    results = txt_results if flag else img_results
    print(
        "The most similar product to your query is: "
        + results["metadatas"][0][0]["Name"]
    )


# PRODUCT RECOMMENDATION (task 2)
def product_recommendation(query, collection_name, output_file, n_res):
    client = chromadb.PersistentClient(
        path=output_file,
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )

    img_collection = client.get_collection(name=collection_name + "_img_collection")
    txt_collection = client.get_collection(name=collection_name + "_txt_collection")

    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    query_embedding = model.get_embeddings(
        contextual_text=query, dimension=DEFAULT_DIM
    ).text_embedding

    txt_results = txt_collection.query(
        query_embeddings=[query_embedding], n_results=n_res
    )
    img_results = img_collection.query(
        query_embeddings=[query_embedding], n_results=n_res
    )

    flag = query_description(query)
    print(f"Top {n_res} recommendations based on the user query are listed as follows:")
    results = txt_results if flag else img_results

    recommended_products = []
    for metadata in results["metadatas"][0]:
        recommended_products.append(metadata)

    return recommended_products


# COMPARISON OF TWO PRODUCTS (task 3)
def generate_query_with_llm(product_name1, product_name2):
    """
    Generates a comparison query for two products using the LLM.
    """
    prompt = (
        f"Compare the following two products:\n1. {product_name1}\n2. {product_name2}\n"
        f"Generate a descriptive query focusing on key differences and similarities, useful for comparison. Keep the prompt < 1024 characters."
    )

    response = llm.generate_content(prompt)
    query = response.text.strip()
    return query


def query_based_similarity_search(flag, query, collection_name, OUTPUT_FILE, n_res):
    """
    Searches for similar items based on the flag indicating if the query is visually or textually descriptive.
    """
    client = chromadb.PersistentClient(
        path=OUTPUT_FILE,
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )

    collection_suffix = "_txt_collection" if flag else "_img_collection"
    collection = client.get_collection(name=collection_name + collection_suffix)

    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    query_embedding = model.get_embeddings(
        contextual_text=query, dimension=DEFAULT_DIM
    ).text_embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_res,
        include=["documents", "metadatas", "distances"],
    )
    return results


def compare_two_products(product_name1, product_name2, collection_name, output_file):
    """
    Compares two products by generating a query, determining if it's text or image-based,
    and then fetching and displaying a comparison table based on similarity results.
    """
    query = generate_query_with_llm(product_name1, product_name2)
    flag = query_description(query)
    results1 = query_based_similarity_search(
        flag, product_name1, collection_name, output_file, 1
    )
    results2 = query_based_similarity_search(
        flag, product_name2, collection_name, output_file, 1
    )
    prompt = (
        f"Compare the following two products based on their key features, similarities, and differences:\n"
        f"1. {product_name1}: {results1['documents']}\n"
        f"2. {product_name2}: {results2['documents']}\n"
        f"Create a comparison table including columns for features, similarities, and differences and then give a solid recommendation what product might be better suited per the user."
    )

    comparison_response = llm.generate_content(prompt)
    comparison_table = comparison_response.text.strip()

    print("Comparison Table:\n", comparison_table)
    return comparison_table

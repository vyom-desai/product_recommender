import json
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from vertexai.vision_models import Image, MultiModalEmbeddingModel
import vertexai
import time

# Initialize Vertex AI
LOCATION = "us-central1"
PROJECT_ID = "applied-ai-practice00"
vertexai.init(project=PROJECT_ID, location=LOCATION)

DIMENSION = 128
RATE_LIMIT = 600
SLEEP_INTERVAL = 60 / RATE_LIMIT


def load_image_from_url(url):
    """
    Loads an image from a URL and converts it into an Image object compatible with Vertex AI.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=5)
    response.raise_for_status()
    return Image(image_bytes=response.content)


def get_text_embedding(model, text, dim):
    """
    Retrieves the text embedding from the model.
    """
    time.sleep(SLEEP_INTERVAL)
    return model.get_embeddings(contextual_text=text, dimension=dim).text_embedding


def get_image_embedding(model, image, text, dim):
    """
    Retrieves the image embedding from the model.
    """
    time.sleep(SLEEP_INTERVAL)
    return model.get_embeddings(
        image=image, contextual_text=text, dimension=dim
    ).image_embedding


def process_instance(instance, model, dim):
    """
    Processes a single instance to generate text and image embeddings.
    """
    text = instance["text"]
    images = instance["images"]
    image_urls = [image["url"] for image in images]
    image_embeddings = []

    try:
        # Generate Text Embedding
        text_embedding = get_text_embedding(model, text, dim)

        # Process each image and get embeddings
        for image_url in image_urls:
            image = load_image_from_url(image_url)
            image_embedding = get_image_embedding(model, image, text, dim)
            image_embeddings.append(image_embedding)
    except Exception as e:
        print(f"Error processing instance: {e}")
        text_embedding = None
        image_embeddings = []

    # Compute the mean of all image embeddings for this instance
    if image_embeddings:
        aggregated_image_embedding = np.mean(image_embeddings, axis=0).tolist()
    else:
        aggregated_image_embedding = None

    return {
        "text_embedding": text_embedding,
        "image_embedding": aggregated_image_embedding,
    }


def gen_files(data, dim, text_output_file, image_output_file):
    """
    Generates text and image embeddings and saves them into separate JSON files.
    """
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    text_results = []
    image_results = []

    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(process_instance, instance, model, dim)
            for item in data
            for instance in item["instances"]
        ]

        embedding_count = 0

        for future in futures:
            result = future.result()
            if result:
                if result["text_embedding"]:
                    text_results.append(result["text_embedding"])
                if result["image_embedding"]:
                    image_results.append(result["image_embedding"])
                print(f"Embedding {embedding_count} processed successfully.")
            embedding_count += 1

    # Save the text embeddings to a JSON file
    with open(text_output_file, "w", encoding="utf-8") as f:
        json.dump(text_results, f, indent=2)
        print(f"Text embeddings saved to: {text_output_file}")

    # Save the image embeddings to a JSON file
    with open(image_output_file, "w", encoding="utf-8") as f:
        json.dump(image_results, f, indent=2)
        print(f"Image embeddings saved to: {image_output_file}")


if __name__ == "__main__":
    start_time = time.time()

    # Load your data
    with open("db_request.json", "r", encoding="utf-8") as file:
        json_data = json.load(file)

    # Generate files
    gen_files(json_data, DIMENSION, "db_text.json", "db_image.json")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Script executed in: {execution_time:.2f} seconds")

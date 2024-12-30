from chroma_db import (
    create_collection,
    similarity_results,
    product_recommendation,
    compare_two_products,
    PERSIST_DIRECTORY,
    COLLECTION_NAME,
    INPUT_FILE,
    N_RECOMMENDATIONS,
)


def main():
    """
    Main function
    """
    create_collection(INPUT_FILE, PERSIST_DIRECTORY, COLLECTION_NAME)
    db_query = ""
    # similarity_results(db_query, COLLECTION_NAME, PERSIST_DIRECTORY, N_RECOMMENDATIONS)
    product_recommendation(
        db_query, COLLECTION_NAME, PERSIST_DIRECTORY, n_res=N_RECOMMENDATIONS
    )
    # compare_two_products(
    #     "Rival Field Messenger",
    #     "Push It Messenger Bag",
    #     COLLECTION_NAME,
    #     PERSIST_DIRECTORY,
    # )


if __name__ == "__main__":
    main()

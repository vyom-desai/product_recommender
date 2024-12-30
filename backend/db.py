"""
Creates a new JSON object and stores it in a JSON file
"""

import json
from google.cloud import bigquery

client = bigquery.Client()
# pylint: disable=invalid-name
db_query = """SELECT * FROM `applied-ai-practice00.prod_dataset.ProductDataset`"""
db_response = client.query(db_query).to_dataframe()
db_json_str = db_response.to_json(orient="records")
db_json_obj = json.loads(db_json_str)
DIMENSION = 128
DEFAULT_EMBEDDING = [0.0] * DIMENSION


def json_to_array(file_path):
    """
    Converts a JSON file to an array
    """
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


db_image = json_to_array("db_image.json")
db_text = json_to_array("db_text.json")


"""
The following json object is the transformed one (including metadata as one of the keys, where the values will be another
list of key-value pairs)
"""
new_json_obj = []
for index, item in enumerate(db_json_obj):
    product_content = f"{item['ID']} {item['Type']} {item['SKU']} {item['Name']} {item['Description']} {item['Price']} {item['Categories']} {item['Images']} {item['Attribute_1_name']} {item['Attribute_1_value']} {item['Attribute_2_name']} {item['Attribute_2_value']} "
    transformed_item = {
        "id": item["ID"],
        "product_content": product_content,
        "metadata": {k: v for k, v in item.items() if (k not in ["ID", "Description"])},
        "image_embedding": db_image[index],
        "text_embedding": db_text[index],
    }
    new_json_obj.append(transformed_item)

with open("db.json", "w", encoding="utf-8") as db_out:
    json.dump(new_json_obj, db_out, indent=2)

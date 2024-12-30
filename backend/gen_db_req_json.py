"""
Creates a new JSON object and stores it in a JSON file
"""

from google.cloud import bigquery
import json
import ast

client = bigquery.Client()
db_query = """
SELECT * FROM `applied-ai-practice00.prod_dataset.ProductDataset`
"""
db_response = client.query(db_query).to_dataframe()
db_json_str = db_response.to_json(orient="records")
db_json_obj = json.loads(db_json_str)
dim = 128

"""
This new json object should satisfy the format of the required JSON object to use the multimodal in order to generate the embeddings
"""
new_json_obj = []
for item in db_json_obj:
    TEXT = f"{item['ID']} {item['Type']} {item['SKU']} {item['Name']} {item['Description']} {item['Price']} {item['Categories']} {item['Images']} {item['Attribute_1_name']} {item['Attribute_1_value']} {item['Attribute_2_name']} {item['Attribute_2_value']} "
    images = item["Images"].split(",")
    image_objects = [{"url": image.strip()} for image in images]
    transformed_item = {
        "instances": [{"images": image_objects, "text": TEXT}],
        "parameters": {"dimension": dim},
    }
    new_json_obj.append(transformed_item)

"""
Store the request JSON object in a JSON file
"""
with open("db_request.json", "w") as db_out:
    json.dump(new_json_obj, db_out, indent=2)

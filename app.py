from flask import Flask, render_template, request, redirect, url_for, session
import google.generativeai as genai
import os
from backend import chroma_db
import json


model = genai.GenerativeModel("gemini-2.0-flash-exp")
my_api_key_gemini = os.getenv("API_KEY")
genai.configure(api_key=my_api_key_gemini)

MAIN_PROMPT = """
You are a conversational AI agent named "Shoppy," working at ShoppersHeaven, a retail store specializing in personalized product recommendations. 
Your primary goal is to assist customers in finding products that match their preferences by gathering relevant details and providing tailored suggestions.
Products include only stuff like clothes (jackets and stuff), and maybe stuff like watches. Here is what you need to know about yourself and your 
responsibilities:

1. **Who You Are**:
   - You are a friendly and knowledgeable AI agent, trained to provide excellent customer service.
   - You represent ShoppersHeaven and are an expert at understanding user preferences and translating them into actionable recommendations.

2. **Your Capabilities**:
   - You can recommend products based on user-provided details such as product category, color preference, price range, and additional attributes like brand or size.
   - If a user provides incomplete information, you will politely ask clarifying questions to fill in the gaps before making recommendations.
   - You are capable of handling multiple rounds of conversation, refining your suggestions based on user feedback and preferences.
   - You keep track of the conversation context to provide consistent and relevant recommendations throughout the session.

3. **How You Operate**:
   - Start each interaction with a warm and friendly greeting to make the user feel welcome.
   - Collect the following details from the user:
     - **Category**: The type of product they’re looking for (e.g., electronics, clothing, furniture).
     - **Color**: Their preferred color or style.
     - **Price Range**: Their budget or price range.
   - If any of these details are missing, ask follow-up questions to gather the required information.
   - Use any additional information (e.g., brand preferences, size, material) to refine your recommendations.
   - Present product recommendations conversationally, ensuring they align with the user's preferences.
   - Ask for feedback after presenting recommendations and refine your suggestions if needed.
   - Conclude the session politely when the user indicates they are done.

4. **Your Personality**:
   - You are polite, enthusiastic, and approachable.
   - Your tone should be conversational and friendly but professional.

5. **How You Handle Tasks**:
   - For every user input, process their preferences and create a structured query for the backend system to fetch the best product recommendations.
   - Incorporate user feedback to improve the accuracy of your recommendations.
   - If the user requests a comparison or additional information about products, ensure your responses are clear and helpful.

When all required information (gender, category, color, price range, and any additional attributes) has been gathered, respond with:
"All information has been gathered: {\"gender\": \"...\", \"category\": \"...\", \"color\": \"...\", \"price_range\": \"...\", \"additional_attributes\": \"...\"}"


Your primary objective is to ensure the user has an enjoyable and productive shopping experience at ShoppersHeaven. Begin every interaction with enthusiasm and a willingness to help.
"""

chat = model.start_chat(history=[])
chat.send_message(MAIN_PROMPT)
app = Flask(__name__)
app.secret_key = os.urandom(24)


@app.errorhandler(404)
def page_not_found(e):
    return redirect(url_for("index"))


COLLECTION_NAME = "db"
PERSIST_DIRECTORY = "./backend/chroma"


def validate_product_response(response):
    """Validate if the response is a list of valid product dictionaries."""
    if isinstance(response, list) and all(isinstance(item, dict) for item in response):
        required_keys = {
            "Name",
            "Price",
            "Attribute_1_value",
            "Attribute_2_value",
            "Images",
        }
        return all(required_keys.issubset(item.keys()) for item in response)
    return False


def format_product_recommendations(product_list):
    """Format product data for chat display with images."""
    formatted_response = "Here are the recommended products:<br><br>"

    for product in product_list:
        name = product.get("Name", "No name available")
        price = product.get("Price", "Price not available")
        size = product.get("Attribute_1_value", "Size not available")
        color = product.get("Attribute_2_value", "Color not available")
        images = product.get("Images", "").split(",")

        formatted_response += f"<strong>Product:</strong> {name}<br>"
        formatted_response += f"<strong>Price:</strong> ${price}<br>"
        formatted_response += f"<strong>Size:</strong> {size}<br>"
        formatted_response += f"<strong>Color:</strong> {color}<br>"

        formatted_response += "<strong>Images:</strong><br>"
        for image_url in images:
            formatted_response += f'<img src="{image_url.strip()}" alt="{name}" style="max-width: 200px; margin: 5px;"><br>'

        formatted_response += "<br>" + "-" * 40 + "<br><br>"

    return formatted_response


@app.route("/", methods=["POST", "GET"])
def index():
    if request.method == "POST":
        try:
            prompt = request.form["prompt"]
            response = chat.send_message(prompt)
            if response.text:
                if "all information has been gathered" in response.text.lower():
                    # Get product recommendations from backend
                    raw_recommendations = chroma_db.product_recommendation(
                        response.text,
                        COLLECTION_NAME,
                        PERSIST_DIRECTORY,
                        chroma_db.N_RECOMMENDATIONS,
                    )

                    # Validate and format recommendations
                    if validate_product_response(raw_recommendations):
                        formatted_recommendations = format_product_recommendations(
                            raw_recommendations
                        )
                        return formatted_recommendations
                    else:
                        return "Sorry, the recommendation data is invalid."

                # Regular chatbot response
                else:
                    return response.text
            else:
                return "Sorry, but I think Gemini didn't want to answer that!"
        except Exception as e:
            return f"Error: {str(e)}"

    return render_template("index.html", **locals())


if __name__ == "__main__":
    app.run(debug=True)

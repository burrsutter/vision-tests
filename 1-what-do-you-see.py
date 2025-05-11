from dotenv import load_dotenv
from openai import OpenAI

import os
import logging
import base64

load_dotenv()

IMAGE_TO_ANALYZE="images/collage-1.png"

INFERENCE_SERVER = os.getenv("INFERENCE_SERVER")
MODEL_NAME = os.getenv("MODEL_NAME")
API_KEY = os.getenv("API_KEY")

print(INFERENCE_SERVER)
print(MODEL_NAME)
print(API_KEY)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

logger.info(f"INFERENCE_SERVER: {INFERENCE_SERVER}")
logger.info(f"MODEL_NAME: {MODEL_NAME}")
logger.info(f"IMAGE_TO_ANALYZE: {IMAGE_TO_ANALYZE}")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode("utf-8")        
        return base64_string


client = OpenAI(
    api_key=API_KEY,
    base_url=INFERENCE_SERVER
    )


messages = [
  {
    "role": "system",
    "content": (
      "You are an image classification system. "
      "Describe the image in detail. "
    )
  },
  {
    "role": "user",
    "content": [
      {"type":"text","text":"Classify this image."},
      {
        "type":"image_url",
        "image_url": {
          "url": f"data:image/jpeg;base64,{encode_image(IMAGE_TO_ANALYZE)}"
        }
      }
    ]
  }
]

response = client.chat.completions.create(
  model=MODEL_NAME,
  messages=messages,
  max_tokens=1000,
  temperature=0.0
)

logger.info(f"RESPONSE: {response.choices[0].message.content}")




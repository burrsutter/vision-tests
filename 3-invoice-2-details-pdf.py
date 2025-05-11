from dotenv import load_dotenv
from openai import OpenAI

import os
import logging
import base64
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError


load_dotenv()

IMAGE_TO_ANALYZE="images/invoice_2.pdf"

INFERENCE_SERVER = os.getenv("INFERENCE_SERVER")
MODEL_NAME = os.getenv("MODEL_NAME")
API_KEY = os.getenv("API_KEY")

OUTPUT_DIR = "images/output/"

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


def convert_pdf_to_png(pdf_path):
    logger.info(f"Converting PDF to PNG: {pdf_path}")
    if not os.path.exists(pdf_path):
        logger.error(f"Error: PDF file not found at {pdf_path}")
        return

    try:
        logger.info(f"Converting {pdf_path} to PNG images...")
        # Convert PDF to a list of PIL images
        images = convert_from_path(pdf_path)

        # Get the base name of the PDF file without extension
        base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        
        output_filename = os.path.join(OUTPUT_DIR, f"{base_filename}.png")
        images[0].save(output_filename, 'PNG')
        logger.info(f"Saved pdf to {output_filename}")

        return output_filename
        

    except PDFInfoNotInstalledError:
        logger.error("Error: pdf2image requires poppler to be installed and in PATH.")
        logger.info("Please install poppler:")
        logger.info("  macOS (brew): brew install poppler")
        logger.info("  Debian/Ubuntu: sudo apt-get install poppler-utils")
        print("  Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases/")
    except PDFPageCountError:
        print(f"Error: Could not get page count for {pdf_path}. Is it a valid PDF?")
    except PDFSyntaxError:
        print(f"Error: PDF file {pdf_path} seems to be corrupted or invalid.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode("utf-8")        
        return base64_string


converted_image = convert_pdf_to_png(IMAGE_TO_ANALYZE)
encoded_image = encode_image(converted_image)


client = OpenAI(
    api_key=API_KEY,
    base_url=INFERENCE_SERVER
    )


# Invoice number
messages = [
  {
    "role": "system",
    "content": (
      "You are an invoice analysis system. "      
    )
  },
  {
    "role": "user",
    "content": [
      {"type":"text","text":"what is the invoice number, only the number"},
      {
        "type":"image_url",
        "image_url": {
          "url": f"data:image/jpeg;base64,{encoded_image}"
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

logger.info(f"Number: {response.choices[0].message.content}")


# Total amount
messages = [
  {
    "role": "system",
    "content": (
      "You are an invoice analysis system. "      
    )
  },
  {
    "role": "user",
    "content": [
      {"type":"text","text":"what is the total amount, only the total"},
      {
        "type":"image_url",
        "image_url": {
          "url": f"data:image/jpeg;base64,{encoded_image}"
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

logger.info(f"Total: {response.choices[0].message.content}")

# Invoice date
messages = [
  {
    "role": "system",
    "content": (
      "You are an invoice analysis system. "      
    )
  },
  {
    "role": "user",
    "content": [
      {"type":"text","text":"what is the invoice date, return only the invoice date"},
      {
        "type":"image_url",
        "image_url": {
          "url": f"data:image/jpeg;base64,{encoded_image}"
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

logger.info(f"Date: {response.choices[0].message.content}")


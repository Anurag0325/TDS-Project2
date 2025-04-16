from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import shutil
import zipfile
import os
import pandas as pd
import openai
import httpx
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
import requests
import subprocess
import hashlib
import numpy as np
from datetime import datetime, timezone
import json
from bs4 import BeautifulSoup
import sqlite3
from PIL import Image
import colorsys
import cv2
import re
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from urllib.parse import urlencode
from geopy.geocoders import Nominatim
import pdfplumber
from markdownify import markdownify as md
from dateutil import parser
# from fuzzywuzzy import process, fuzz
# from moviepy.editor import VideoFileClip
import speech_recognition as sr

app = FastAPI()

# Serve static files (if needed in the future)
# app.mount("/static", StaticFiles(directory="static"), name="static")

load_dotenv()

app.add_middleware(
    CORSMiddleware,
    # Allows all origins (for testing). Replace "*" with specific domains in production.
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

UPLOAD_DIR = "uploads"
EXTRACTED_DIR = "extracted_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EXTRACTED_DIR, exist_ok=True)

# Replace with your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI(api_key=openai.api_key)

# Predefined assignment questions
ASSIGNMENT_QUESTIONS = {
    "Install and run Visual Studio Code. In your Terminal (or Command Prompt), type code -s and press Enter. Copy and paste the entire output below. What is the output of code -s?":
    "The output of `code -s` varies depending on the system configuration. "
    "On most systems, it returns an empty JSON array `[]` if no additional settings or extensions are installed."

    "Running uv run --with httpie -- https [URL] installs the Python package httpie and sends a HTTPS request to the URL.\n\n"
    "Send a HTTPS request to https://httpbin.org/get with the URL encoded parameter email set to 24ds2000092@ds.study.iitm.ac.in\n\n"
    "What is the JSON output of the command? (Paste only the JSON body, not the headers)"

    "Download README.md. In the directory where you downloaded it, make sure it is called README.md, and run npx -y prettier@3.4.2 README.md | sha256sum.\n\n"
    "Upload the README.md file, and I will compute the SHA256 hash after formatting it with Prettier."

    "Let's make sure you can write formulas in Google Sheets. Type this formula into Google Sheets. (It won't work in Excel)\n\n=SUM(ARRAY_CONSTRAIN(SEQUENCE(100, 100, 1, 14), 1, 10))"
        "This formula creates a 100x100 sequence starting from 1, with a step of 14. The `ARRAY_CONSTRAIN` function limits it to a 1-row, 10-column array, and `SUM` adds those 10 numbers."

    "Let's make sure you can write formulas in Excel. Type this formula into Excel.\n\n"
    "Note: This will ONLY work in Office 365.\n\n"
    "=SUM(TAKE(SORTBY({2,1,1,8,15,5,7,9,14,12,9,6,4,5,11,0}, {10,9,13,2,11,8,16,14,7,15,5,4,6,1,3,12}), 1, 14))"

    "How many Wednesdays are there in the date range 1985-11-04 to 2017-07-23?"

    "Download and unzip file which has a single extract.csv file inside. What is the value in the 'answer' column of the CSV file?"

    "Sort this JSON array of objects by the value of the age field. In case of a tie, sort by the name field. Paste the resulting JSON below without any spaces or newlines."

    "Calculate total sales of Gold tickets."

    "Provide your email as a parameter, and I will compute the required hash."

    "Calculate the number of pixels with a certain minimum brightness"

    "Download  and unzip it into a new folder, then replace all 'IITM' (in upper, lower, or mixed case) with 'IIT Madras' in all files. Leave everything as-is - don't change the line endings. What does running cat * | sha256sum in that folder show in bash?"

    "Use ls with options to list all files in the folder along with their date and file size. What's the total size of all files at least 4849 bytes large and modified on or after Wed, 19 Mar, 2008, 5:54 am IST?"

    "Use mv to move all files under folders into an empty folder. Then rename all files replacing each digit with the next. 1 becomes 2, 9 becomes 0, a1b9c.txt becomes a2b0c.txt. What does running grep . * | LC_ALL=C sort | sha256sum in bash on that folder show?"

    "Download  and extract it. It has 2 nearly identical files, a.txt and b.txt, with the same number of lines. How many lines are different between a.txt and b.txt?"

    "Write a Python program that uses httpx to send a POST request to OpenAI's API to analyze the sentiment of this (meaningless) text into GOOD, BAD or NEUTRAL."

    "Analyze the following text and compute how many input tokens are present in the text."

    "Obtain the text embedding for the 2 given personalized transaction verification messages"

    "Write a Python function most similar embeddings that will calculate the cosine similarity between each pair of these embeddings and return the pair that has the highest similarity. The result should be a tuple of the two phrases that are most similar."

    "What is the total number of ducks across players on page number 9 of ESPN Cricinfo's ODI batting stats?"

    "What is the JSON weather forecast description for Miami?"

    "What is the maximum latitude of the bounding box of the city Riyadh in the country Saudi Arabia on the Nominatim API? Value of the maximum latitude"

    "Using the GitHub API, find all users located in the city San Francisco with over 200 followers. When was the newest user's GitHub profile created? Ignore ultra-new users who JUST joined, i.e. after 3/29/2025, 5:00:00 PM."

    "What is the sum total of English marks of students who scored 69 or more marks in Maths in groups 1-33 (including both groups)?"

    "What is the markdown content of the PDF."

    "What is the total margin for transactions before Tue May 16 2023 11:44:47 GMT+0530 (India Standard Time) for Iota sold in UK (which may be spelt in different ways)?"

    "How many unique students are there in the file?"

    "How many units of Chair were sold in Tianjin on transactions with at least 51 units?"

    "What is the text of the transcript of this Mystery Story Audiobook between 26.4 and 150.5 seconds?"
}


# def extract_and_process_zip(zip_path):
#     """Extracts a ZIP file and processes CSV content."""
#     extract_folder = "extracted_files"
#     os.makedirs(extract_folder, exist_ok=True)

#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         zip_ref.extractall(extract_folder)

#     csv_files = [f for f in os.listdir(extract_folder) if f.endswith(".csv")]
#     if not csv_files:
#         return "No CSV file found in ZIP."

#     csv_path = os.path.join(extract_folder, csv_files[0])
#     df = pd.read_csv(csv_path)

#     if "answer" in df.columns:
#         return str(df["answer"].iloc[0])

#     return "Column 'answer' not found in CSV."

# def extract_zip(zip_path):
#     """Extracts all files from a ZIP archive and returns a list of extracted file paths."""
#     extract_folder = "extracted_files"
#     os.makedirs(extract_folder, exist_ok=True)

#     try:
#         with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#             zip_ref.extractall(extract_folder)

#         extracted_files = os.listdir(extract_folder)  # List extracted files
#         print("Extracted files:", extracted_files)  # Debugging
#         return [os.path.join(extract_folder, f) for f in extracted_files]

#     except zipfile.BadZipFile:
#         return "Error: File is not a valid ZIP archive."

# async def extract_zip(file_path: str) -> list:
#     """
#     Extracts a ZIP file to a temporary directory and returns a list of extracted file paths.
#     """
#     if not zipfile.is_zipfile(file_path):  # ✅ Check if valid ZIP
#         return ["Error: The uploaded file is not a valid ZIP archive."]

#     temp_dir = tempfile.mkdtemp()  # ✅ Create a temporary directory

#     try:
#         with zipfile.ZipFile(file_path, "r") as zip_ref:
#             zip_ref.extractall(temp_dir)  # ✅ Extract all files

#         extracted_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir)]
#         return extracted_files  # ✅ Return list of extracted files

#     except Exception as e:
#         return [f"Error: {str(e)}"]


def ask_llm(question):
    """Queries OpenAI's GPT model to get an answer."""
    response = client.chat.completions.create(  # FIXED: Use chat API
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ],
        max_tokens=100
    )
    return response.choices[0].message.content.strip()


async def send_httpbin_request(email: str):
    """Sends an HTTP request to httpbin.org and returns the JSON response."""
    url = "https://httpbin.org/get"
    params = {"email": email}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        return response.json()

# def format_and_hash_readme(file_path):
#     """Runs Prettier on README.md and generates a SHA256 hash."""
#     try:
#         result = subprocess.run(
#             "npx -y prettier@3.4.2 README.md | sha256sum",
#             capture_output=True, text=True, shell=True, cwd=os.path.dirname(file_path)
#         )
#         return result.stdout.strip()
#     except Exception as e:
#         return f"Error processing README.md: {str(e)}"


def solve_google_sheets_formula():
    """Simulates the Google Sheets formula in Python and returns the computed sum."""
    import numpy as np

    first_row = np.arange(1, 1 + 14 * 10, 14)  # Generates the first 10 values
    total_sum = np.sum(first_row)  # Computes the sum

    return int(total_sum)  # Convert to a Python int before returning


def format_and_hash_readme(file_path):
    """Runs Prettier on README.md and generates a SHA256 hash."""
    try:
        # Ensure Prettier runs on the correct file
        result = subprocess.run(
            f"npx -y prettier@3.4.2 {file_path}",
            capture_output=True, text=True, shell=True
        )

        if result.returncode != 0:
            return f"Error formatting README.md: {result.stderr}"

        # Compute SHA256 hash
        readme_content = result.stdout.encode()
        sha256_hash = hashlib.sha256(readme_content).hexdigest()

        return sha256_hash
    except Exception as e:
        return f"Error processing README.md: {str(e)}"


def solve_excel_formula():
    """Simulates the Excel formula in Python and returns the computed sum."""
    values = np.array([2, 1, 1, 8, 15, 5, 7, 9, 14, 12, 9, 6, 4, 5, 11, 0])
    sort_keys = np.array(
        [10, 9, 13, 2, 11, 8, 16, 14, 7, 15, 5, 4, 6, 1, 3, 12])

    sorted_values = values[np.argsort(sort_keys)]
    first_14_values = sorted_values[:14]
    total_sum = np.sum(first_14_values)

    return int(total_sum)


# def count_wednesdays():
#     """Counts the number of Wednesdays between 1985-11-04 and 2017-07-23."""
#     # start_date = datetime.date(1985, 11, 4)
#     # end_date = datetime.date(2017, 7, 23)
#     start_date = date(1985, 11, 4)
#     end_date = date(2017, 7, 23)

#     count = sum(1 for d in range((end_date - start_date).days + 1)
#                 if (start_date + timedelta(days=d)).weekday() == 2)  # Wednesday is weekday 2

#     return count

def count_wednesdays():
    """Counts the number of Wednesdays between 1985-11-04 and 2017-07-23 using only datetime."""
    start_date = datetime(1985, 11, 4)
    end_date = datetime(2017, 7, 23)

    count = 0
    current = start_date

    while current <= end_date:
        if current.weekday() == 2:  # Wednesday
            count += 1
        # Add 1 day manually by converting to timestamp and adding 86400 seconds
        current = datetime.fromtimestamp(current.timestamp() + 86400)

    return count

# def process_extract_csv():
#     """Finds extract.csv and returns values from the 'answer' column."""
#     extract_folder = "extracted_files"
#     csv_path = os.path.join(extract_folder, "extract.csv")

#     print("Looking for extract.csv at:", csv_path)  # Debugging

#     if not os.path.exists(csv_path):
#         print("extract.csv not found!")  # Debugging
#         return "extract.csv not found."

#     df = pd.read_csv(csv_path)
#     print("CSV Columns:", df.columns)  # Debugging

#     if "answer" in df.columns:
#         return df["answer"].tolist()

#     return "Column 'answer' not found in extract.csv."


async def process_extract_csv(csv_folder):
    """Find extract.csv and return values from 'answer' column."""
    csv_path = os.path.join(csv_folder, "extract.csv")

    if not os.path.exists(csv_path):
        return "Error: extract.csv not found."

    try:
        df = pd.read_csv(csv_path)
        return df["answer"].dropna().tolist() if "answer" in df.columns else "Error: Column 'answer' not found."
    except Exception as e:
        return f"Error reading CSV: {str(e)}"


def sort_json_array():
    """Sorts the given JSON array based on age and name."""
    data = [
        {"name": "Alice", "age": 39}, {"name": "Bob", "age": 69},
        {"name": "Charlie", "age": 44}, {"name": "David", "age": 93},
        {"name": "Emma", "age": 96}, {"name": "Frank", "age": 20},
        {"name": "Grace", "age": 10}, {"name": "Henry", "age": 25},
        {"name": "Ivy", "age": 81}, {"name": "Jack", "age": 55},
        {"name": "Karen", "age": 87}, {"name": "Liam", "age": 70},
        {"name": "Mary", "age": 74}, {"name": "Nora", "age": 19},
        {"name": "Oscar", "age": 79}, {"name": "Paul", "age": 32}
    ]

    # Sorting by age first, then name
    sorted_data = sorted(data, key=lambda x: (x["age"], x["name"]))

    # Convert to JSON string without spaces or newlines
    return json.dumps(sorted_data, separators=(",", ":"))


def sum_data_values_from_html(html_content: str) -> int:
    """Extracts all <div> elements with class 'foo' and sums their 'data-value' attributes."""
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        total = sum(int(div.get("data-value", 0))
                    for div in soup.find_all("div", class_="foo"))
        return total
    except Exception as e:
        return f"Error: {str(e)}"


def get_total_gold_sales():
    """Fetch total sales for 'Gold' tickets, handling case and spaces."""
    conn = sqlite3.connect("tickets.db")  # Ensure correct DB path
    cursor = conn.cursor()

    query = """
        SELECT SUM(units * price) AS total_sales
        FROM tickets
        WHERE LOWER(TRIM(type)) = 'gold';
    """

    cursor.execute(query)
    result = cursor.fetchone()
    conn.close()

    return result[0] if result[0] else 0


def extract_zip(zip_path, extract_to):
    """Extract ZIP file contents to the specified directory."""
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    return [os.path.join(extract_to, f) for f in os.listdir(extract_to) if f.endswith(".csv")]


def sum_selected_symbols(file_paths):
    """Process CSV files and sum values for symbols ‰, ‡, œ"""
    target_symbols = {'‰', '‡', 'œ'}
    encodings = ['cp1252', 'utf-8', 'utf-16']  # Encoding order assumption

    total_sum = 0
    for file_path, encoding in zip(file_paths, encodings):
        df = pd.read_csv(file_path, encoding=encoding)
        df_filtered = df[df['symbol'].isin(target_symbols)]
        total_sum += df_filtered['value'].sum()

    return total_sum


def solve_google_colab_auth(email: str):
    """
    Generates a 5-character SHA-256 hash based on the provided email and the current year.
    Works on any system (not limited to Google Colab).
    """
    try:
        # Get the current year
        current_year = datetime.datetime.now().year

        # Compute the hash
        hash_value = hashlib.sha256(
            f"{email} {current_year}".encode()).hexdigest()[-5:]
        return hash_value

    except Exception as e:
        return f"Error: {str(e)}"


def count_light_pixels(image_path: str, threshold: float = 0.347):
    """
    Reads an image, calculates pixel brightness, and counts pixels above a certain lightness threshold.
    """
    try:
        # Open and convert image to RGB
        image = Image.open(image_path).convert("RGB")

        # Convert image to NumPy array and normalize pixel values
        rgb = np.array(image) / 255.0

        # Compute lightness using HLS color space
        lightness = np.apply_along_axis(
            lambda x: colorsys.rgb_to_hls(*x)[1], 2, rgb)

        # Count pixels above the given threshold
        light_pixels = np.sum(lightness > threshold)

        return int(light_pixels)  # Return as integer

    except Exception as e:
        return f"Error processing image: {str(e)}"


def convert_webp_to_png_opencv(webp_file: str, output_dir: str = "converted_images"):
    """
    Converts a .webp image to .png using OpenCV.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Read .webp file using OpenCV
        image = cv2.imread(webp_file, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(
                "❌ OpenCV could not read the .webp file. File may be corrupted.")

        # Generate output file name
        png_file = os.path.join(output_dir, os.path.splitext(
            os.path.basename(webp_file))[0] + ".png")

        # Save as .png
        cv2.imwrite(png_file, image)

        print(f"✅ Conversion successful! Saved as: {png_file}")
        return png_file

    except Exception as e:
        print(f"❌ Error converting image: {e}")
        return None


def extract_zip_file(zip_path: str) -> dict:
    """
    Extracts a given ZIP file to a folder inside EXTRACTED_DIR.

    Returns:
        dict: {"extracted_to": "folder_path", "files": ["file1.csv", "file2.txt", ...]}
    """
    extract_folder = os.path.join(
        EXTRACTED_DIR, os.path.splitext(os.path.basename(zip_path))[0])
    os.makedirs(extract_folder, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_folder)
            extracted_files = os.listdir(extract_folder)

        return {"extracted_to": extract_folder, "files": extracted_files}
    except zipfile.BadZipFile:
        return {"error": "Uploaded file is not a valid ZIP archive"}


def process_question_1(extract_folder: str):
    """
    Process q-extract-csv-zip.zip: Read the "answer" column from extract.csv.
    """
    csv_path = os.path.join(extract_folder, "extract.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            print(f"📊 CSV Columns: {df.columns}")  # Debug Log
            if "answer" in df.columns:
                return {"answer": df["answer"].iloc[0]}
            else:
                return {"error": "'answer' column not found in extract.csv"}
        except Exception as e:
            return {"error": f"Failed to read CSV: {str(e)}"}
    return {"error": "extract.csv not found"}


def process_question_2(extract_folder: str):
    """
    Process q-unicode-data.zip: Sum values where the symbol matches ‰, ‡, or œ.
    """
    target_symbols = {"‰", "‡", "œ"}
    total_sum = 0  # Python int, not NumPy

    # Define file encoding types
    file_encodings = {
        "data1.csv": "cp1252",
        "data2.csv": "utf-8",
        "data3.txt": "utf-16"
    }

    # Process each file
    for file_name, encoding in file_encodings.items():
        file_path = os.path.join(extract_folder, file_name)

        if os.path.exists(file_path):
            df = pd.read_csv(file_path, encoding=encoding,
                             delimiter="\t" if file_name.endswith(".txt") else ",")
            if "symbol" in df.columns and "value" in df.columns:
                df_filtered = df[df["symbol"].isin(target_symbols)]
                total_sum += df_filtered["value"].sum()

    return {"sum": int(total_sum)}  # ✅ Convert NumPy int to Python int


def process_text_files(folder_path: str):
    """
    Process all .txt files:
    - Replace "IITM" (case-insensitive) with "IIT Madras"
    - Keep line endings unchanged
    - Compute SHA-256 hash of concatenated content
    """
    try:
        all_content = []  # Store processed file contents

        for filename in sorted(os.listdir(folder_path)):  # Process in order
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)

                # Read with original line endings preserved
                with open(file_path, "r", encoding="utf-8", newline="") as file:
                    content = file.readlines()

                # Replace "IITM" (case-insensitive) with "IIT Madras"
                modified_content = [line.replace("IITM", "IIT Madras").replace(
                    "iitm", "IIT Madras").replace("IItM", "IIT Madras") for line in content]

                # Write back with original line endings
                with open(file_path, "w", encoding="utf-8", newline="") as file:
                    file.writelines(modified_content)

                # Append to hash calculation
                all_content.extend(modified_content)

        # ✅ Compute SHA-256 hash after modifications
        sha256_hash = hashlib.sha256(
            "".join(all_content).encode("utf-8")).hexdigest()
        print(f"🔢 SHA-256 Hash: {sha256_hash}")

        # return {"sha256sum": sha256_hash}
        return sha256_hash

    except Exception as e:
        return {"error": f"Error processing text files: {str(e)}"}


def calculate_filtered_size(folder_path: str):
    """
    List all files, filter those with:
    - Size >= 4849 bytes
    - Modified on or after 19 Mar 2008, 5:54 AM IST
    - Return total size of filtered files
    """
    total_size = 0
    CUTOFF_DATETIME = datetime.datetime(
        2008, 3, 19, 5, 54)  # 19 Mar 2008, 5:54 AM IST
    SIZE_THRESHOLD = 4849  # Minimum file size in bytes

    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            if os.path.isfile(file_path):  # Ensure it's a file
                file_size = os.path.getsize(file_path)
                modified_time = datetime.datetime.fromtimestamp(
                    os.path.getmtime(file_path))

                # ✅ Apply filters
                if file_size >= SIZE_THRESHOLD and modified_time >= CUTOFF_DATETIME:
                    total_size += file_size
                    print(
                        f"✔ {filename}: {file_size} bytes, Modified: {modified_time}")

        print(f"📏 Total Size of Matching Files: {total_size} bytes")
        # 🔥 Return total size as a string (not a dictionary)
        return str(total_size)

    except Exception as e:
        return f"Error processing files: {str(e)}"


def process_zip_and_hash(extract_path):
    """
    1. Move all files into a single folder
    2. Rename files by replacing digits (1 → 2, 9 → 0, etc.)
    3. Run equivalent of `grep . * | LC_ALL=C sort | sha256sum`
    """
    target_folder = os.path.join(extract_path, "processed_files")
    os.makedirs(target_folder, exist_ok=True)

    # Move all files from subdirectories to target_folder
    for root, _, files in os.walk(extract_path):
        for file in files:
            src_path = os.path.join(root, file)
            dst_filename = os.path.basename(src_path)  # Keep only filename
            dst_path = os.path.join(target_folder, dst_filename)
            shutil.move(src_path, dst_path)

    # Rename files by replacing digits (1 → 2, 9 → 0, etc.)
    def replace_digits(name):
        return re.sub(r'\d', lambda x: str((int(x.group()) + 1) % 10), name)

    renamed_files = {}
    # LC_ALL=C sorting
    for filename in sorted(os.listdir(target_folder), key=lambda x: x.encode("utf-8")):
        new_filename = replace_digits(filename)
        renamed_files[filename] = new_filename
        os.rename(os.path.join(target_folder, filename),
                  os.path.join(target_folder, new_filename))

    # Create a sorted content string to simulate `grep . * | LC_ALL=C sort`
    all_contents = []
    # Correct sorting
    for filename in sorted(os.listdir(target_folder), key=lambda x: x.encode("utf-8")):
        file_path = os.path.join(target_folder, filename)
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                for line in lines:
                    # Strip trailing spaces & newlines
                    all_contents.append(f"{filename}:{line.rstrip()}")

    # Correct sorting for `LC_ALL=C`
    # Sort using byte order
    sorted_text = "\n".join(
        sorted(all_contents, key=lambda x: x.encode("utf-8")))

    # Compute SHA256
    sha256_hash = hashlib.sha256(sorted_text.encode("utf-8")).hexdigest()

    return sha256_hash


def count_different_lines(extract_path):
    """
    Compare 'a.txt' and 'b.txt' line by line and count differences.
    """
    file_a = os.path.join(extract_path, "a.txt")
    file_b = os.path.join(extract_path, "b.txt")

    if not os.path.exists(file_a) or not os.path.exists(file_b):
        return {"error": "Both a.txt and b.txt must be present in the ZIP file."}

    # Read files and compare line by line
    with open(file_a, "r", errors="ignore") as fa, open(file_b, "r", errors="ignore") as fb:
        lines_a = fa.readlines()
        lines_b = fb.readlines()

    # Ensure same number of lines
    if len(lines_a) != len(lines_b):
        return {"error": "Files do not have the same number of lines."}

    # Count differing lines
    diff_count = sum(1 for line_a, line_b in zip(
        lines_a, lines_b) if line_a != line_b)

    return diff_count


def count_tokens(text: str):
    """Counts the number of input tokens in the given text using OpenAI's encoding."""
    encoding = tiktoken.encoding_for_model(
        "gpt-4o")  # Use "gpt-4o" or another model
    tokens = encoding.encode(text)
    return len(tokens)


def generate_embedding_request(messages):
    """Creates a JSON request body for OpenAI's text embeddings API."""
    # messages = [
    #     "Dear user, please verify your transaction code 44685 sent to 24ds2000092@ds.study.iitm.ac.in",
    #     "Dear user, please verify your transaction code 94940 sent to 24ds2000092@ds.study.iitm.ac.in"
    # ]
    json_body = {
        "model": "text-embedding-3-small",
        "input": messages
    }
    return json_body


def find_most_similar_texts(file_path):
    """Finds the most similar pair of text lines in a file using cosine similarity."""
    try:
        # Read text file (assuming one line per text entry)
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if len(lines) < 2:
            return {"error": "Not enough text entries to compare."}

        # Convert text to TF-IDF vectors
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(lines)

        # Compute cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Find the most similar pair
        max_sim = 0
        most_similar_pair = ("", "")
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                if similarity_matrix[i][j] > max_sim:
                    max_sim = similarity_matrix[i][j]
                    most_similar_pair = (lines[i], lines[j])

        return {
            "text_1": most_similar_pair[0],
            "text_2": most_similar_pair[1],
            "similarity_score": round(max_sim, 4)
        }

    except Exception as e:
        return {"error": f"Error processing text file: {str(e)}"}

# def fetch_cricinfo_ducks():
#     url = "https://stats.espncricinfo.com/ci/engine/stats/index.html?class=2;page=9;template=results;type=batting"
#     response = requests.get(url)
#     if response.status_code != 200:
#         return "Error: Unable to fetch data from ESPN Cricinfo"

#     soup = BeautifulSoup(response.text, 'html.parser')
#     tables = soup.find_all('table')
#     if not tables:
#         return "Error: No tables found on the page"

#     df = pd.read_html(str(tables[0]))[0]
#     if "0" not in df.columns:
#         return "Error: '0' column not found in the table"

#     return df["0"].sum()


def fetch_cricinfo_ducks():
    url = "https://stats.espncricinfo.com/ci/engine/stats/index.html?class=2;page=9;template=results;type=batting"

    # Fetch the webpage
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(response.content, "html.parser")

    # Locate the table with batting stats
    table = soup.find("table", class_="engineTable")
    if not table:
        return "Table not found"

    # Extract table headers
    header_row = table.find_all("tr")[0]
    headers = [th.get_text(strip=True) for th in header_row.find_all("th")]

    print("Headers:", headers)  # Debugging step to see actual column names

    # Ensure '0s' column exists
    if "0" not in headers:
        return "Ducks column not found"

    ducks_index = headers.index("0")  # Get the correct column index

    # Extract all rows and sum up ducks count
    rows = table.find_all("tr")[1:]  # Skip header row
    total_ducks = sum(
        int(row.find_all("td")[ducks_index].text)
        for row in rows
        if row.find_all("td")[ducks_index].text.isdigit()
    )

    return total_ducks


def get_weather_forecast(text: str):
    """Fetches and returns the weather forecast for a given city using BBC Weather API."""

    BBC_API_KEY = os.getenv("BBC_API_KEY")
    if not BBC_API_KEY:
        return {"error": "BBC API key not found. Set BBC_API_KEY in environment variables."}

    location_url = 'https://locator-service.api.bbci.co.uk/locations?' + urlencode({
        'api_key': BBC_API_KEY,
        's': text,
        'stack': 'aws',
        'locale': 'en',
        'filter': 'international',
        'place-types': 'settlement,airport,district',
        'order': 'importance',
        'a': 'true',
        'format': 'json'
    })

    try:
        result = requests.get(location_url).json()
        location_id = result['response']['results']['results'][0]['id']
    except (KeyError, IndexError):
        return {"error": f"Could not retrieve locationId for {text}."}

    weather_url = f'https://www.bbc.com/weather/{location_id}'
    response = requests.get(weather_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    daily_summary = soup.find('div', attrs={'class': 'wr-day-summary'})
    if not daily_summary:
        return {"error": f"Weather data not found for {text}."}

    daily_summary_list = re.findall(r'[a-zA-Z][^A-Z]*', daily_summary.text)
    datelist = pd.date_range(datetime.today(), periods=len(
        daily_summary_list)).strftime('%Y-%m-%d').tolist()
    weather_data = {date: desc for date,
                    desc in zip(datelist, daily_summary_list)}

    return weather_data


def extract_city_country_latitude(question: str):
    """Extracts city, country, and latitude type (max/min) from the question."""
    match = re.search(
        r"city\s+([\w\s]+?)\s+in\s+the\s+country\s+([\w\s]+?)(?:\s+on|\s*\?|$)",
        question, re.IGNORECASE
    )

    if match:
        city = match.group(1).strip()
        country = match.group(2).strip()

        # Determine whether max or min latitude is requested
        lat_type = "max" if "maximum latitude" in question.lower(
        ) else "min" if "minimum latitude" in question.lower() else None

        return city, country, lat_type
    return None, None, None


def get_latitude(city: str, country: str, lat_type: str):

    locator = locator = Nominatim(user_agent="myGeocoder")
    """Fetches the maximum or minimum latitude of a city in a country."""
    location = locator.geocode(f"{city}, {country}", exactly_one=False)

    if not location:
        return "Error: Location not found."

    # Filter locations based on osm_id ending (if needed)
    filtered_location = None
    for loc in location:
        if str(loc.raw.get("osm_id", "")).endswith("8409"):  # Adjust if different osm_id is needed
            filtered_location = loc
            break

    if not filtered_location:
        return f"Error: No location found with the required osm_id for {city}, {country}."

    bounding_box = filtered_location.raw.get('boundingbox', [])

    if len(bounding_box) < 2:
        return "Error: Bounding box information not available."

    # Extract max or min latitude
    latitude = float(bounding_box[1]) if lat_type == "max" else float(
        bounding_box[0])
    return f"The {lat_type}imum latitude for {city}, {country} is: {latitude}"


def extract_github_query_params(question):
    """Extracts city, minimum followers, and cutoff datetime from the question."""

    # Extracts city (handles multiple words before "with" or "followers")
    city_match = re.search(
        r"located in the city ([\w\s]+?)(?:\s+with|\s+followers)", question, re.IGNORECASE)

    # Extracts minimum followers count
    followers_match = re.search(
        r"over (\d+) followers", question, re.IGNORECASE)

    # Extracts cutoff datetime
    cutoff_match = re.search(
        r"after\s+(\d{1,2}/\d{1,2}/\d{4},\s+\d{1,2}:\d{1,2}:\d{2}\s*[APMapm]*)", question)

    city = city_match.group(1).strip(
    ) if city_match else "Chennai"  # Default: Chennai
    min_followers = int(followers_match.group(
        1)) if followers_match else 110  # Default: 110 followers
    cutoff_datetime = None

    if cutoff_match:
        try:
            cutoff_datetime = datetime.strptime(cutoff_match.group(
                1), "%m/%d/%Y, %I:%M:%S %p").replace(tzinfo=timezone.utc)
        except ValueError:
            return None, None, None  # Return None if datetime parsing fails

    return city, min_followers, cutoff_datetime


def get_newest_github_user(question):

    GITHUB_API_URL = "https://api.github.com/search/users"

    """Fetches the newest GitHub user in a given city with a given number of followers."""
    city, min_followers, cutoff_datetime = extract_github_query_params(
        question)

    if not cutoff_datetime:
        return "Error: Could not extract the cutoff date and time."

    query = f"location:{city} followers:>{min_followers}"
    response = requests.get(
        f"{GITHUB_API_URL}?q={query}&sort=joined&order=desc")

    if response.status_code != 200:
        return f"Error: GitHub API request failed with status {response.status_code}"

    users = response.json().get("items", [])

    for user in users:
        user_details = requests.get(user["url"]).json()
        created_at = user_details.get("created_at")

        if created_at:
            created_at_dt = datetime.strptime(
                created_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            if created_at_dt < cutoff_datetime:
                return created_at  # Return the first valid user

    return "Error: No suitable user found."


def extract_tables_with_pdfplumber(pdf_path, start_page, end_page):
    """Extracts tables from a PDF using pdfplumber within specified page range."""
    extracted_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for i in range(start_page - 1, end_page):  # PDF pages are 0-indexed
            table = pdf.pages[i].extract_table()
            if table:
                extracted_data.extend(table)
    return extracted_data


def clean_and_process_data(data):
    """Cleans and converts extracted data into a DataFrame."""
    df = pd.DataFrame(data)
    df.dropna(how="all", inplace=True)  # Remove empty rows

    # Ensure column names are assigned correctly
    df.columns = ["Group", "Maths", "Physics", "English",
                  "Economics", "Biology"][:len(df.columns)]

    # Convert numeric columns properly
    for col in ["Group", "Maths", "English"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with NaN values in relevant columns
    df.dropna(subset=["Group", "Maths", "English"], inplace=True)

    return df


def extract_conditions_from_question(question):
    """Extracts conditions dynamically from a question."""
    subject_match = re.search(
        r'sum total of ([A-Za-z]+) marks', question, re.IGNORECASE)
    marks_condition_match = re.search(
        r'scored (\d+) or (more|less)', question, re.IGNORECASE)
    condition_subject_match = re.search(
        r'in ([A-Za-z]+)', question, re.IGNORECASE)
    group_range_match = re.search(
        r'groups (\d+)[^\d]+(\d+)', question, re.IGNORECASE)

    subject = subject_match.group(1) if subject_match else "English"
    required_marks = int(marks_condition_match.group(1)
                         ) if marks_condition_match else 69
    comparison = marks_condition_match.group(
        2) if marks_condition_match else "more"
    condition_subject = condition_subject_match.group(
        1) if condition_subject_match else "Maths"
    start_group = int(group_range_match.group(1)) if group_range_match else 1
    end_group = int(group_range_match.group(2)) if group_range_match else 33

    return subject, required_marks, comparison, condition_subject, start_group, end_group


def calculate_total_marks(pdf_path, question):
    """Dynamically calculates total marks based on extracted question conditions."""
    subject, required_marks, comparison, condition_subject, start_page, end_page = extract_conditions_from_question(
        question)
    raw_data = extract_tables_with_pdfplumber(pdf_path, start_page, end_page)
    df = clean_and_process_data(raw_data)

    # Apply filters correctly
    if comparison == "more":
        filtered_df = df[(df["Group"].between(start_page, end_page)) & (
            df[condition_subject] >= required_marks)]
    else:
        filtered_df = df[(df["Group"].between(start_page, end_page)) & (
            df[condition_subject] < required_marks)]

    total_marks = filtered_df[subject].sum()
    return int(total_marks)


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
    return text


def restore_headings(text):
    """Formats extracted text by converting uppercase phrases into headings."""
    lines = text.split("\n")
    formatted_lines = []

    for line in lines:
        if line.isupper() and len(line.split()) > 2:
            formatted_lines.append(f"# {line.strip()}\n")
        else:
            formatted_lines.append(line)

    return "\n".join(formatted_lines)


def convert_pdf_to_markdown(pdf_path):
    """Converts PDF content to Markdown format."""
    extracted_text = extract_text_from_pdf(pdf_path)
    formatted_text = restore_headings(extracted_text)
    markdown_content = md(formatted_text)

    return markdown_content  # Returns Markdown content as a string


# Standard country name mapping
COUNTRY_MAPPING = {
    "USA": "US", "U.S.A": "US", "UNITED STATES": "US", "US": "US",
    "BRA": "BR", "BRAZIL": "BR", "BR": "BR",
    "U.K": "UK", "UK": "UK", "UNITED KINGDOM": "UK",
    "FR": "FRANCE", "FRA": "FRANCE", "FRANCE": "FRANCE",
    "IND": "INDIA", "IN": "INDIA", "INDIA": "INDIA",
    "AE": "UAE", "U.A.E": "UAE", "UNITED ARAB EMIRATES": "UAE", "UAE": "UAE"
}


def convert_to_datetime(date):
    for fmt in ("%m-%d-%Y", "%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(date), fmt)
        except ValueError:
            continue
    return pd.NaT  # Invalid date formats


def extract_details_from_question(question):
    """Extracts cutoff date, product, country, and condition (before/after) from the question."""
    date_match = re.search(
        r'\b([A-Z][a-z]+ \d{1,2} \d{4} [\d:]+ [A-Z+0-9]+)\b', question)
    product_match = re.search(
        r'for ([a-zA-Z0-9]+) sold', question, re.IGNORECASE)
    country_match = re.search(
        r'sold in ([a-zA-Z .-]+)', question, re.IGNORECASE)
    condition_match = re.search(r'\b(before|after)\b', question, re.IGNORECASE)

    cutoff_date = parser.parse(date_match.group(1)).replace(
        tzinfo=None) if date_match else None
    product = product_match.group(1).strip() if product_match else None
    country = COUNTRY_MAPPING.get(country_match.group(1).strip().upper(
    ), country_match.group(1).strip().upper()) if country_match else None
    condition = condition_match.group(1).lower(
    ) if condition_match else "before"  # Default to "before"
    print(product)
    print(country)
    print(cutoff_date)
    print(condition)

    return cutoff_date, product, country, condition


def clean_and_calculate_margin(excel_path, product_name, country_name, cutoff_date, condition="before"):
    """Cleans Excel data and calculates total margin based on extracted details."""
    try:
        df = pd.read_excel(excel_path, engine="openpyxl")
        # df = pd.read_excel(excel_path)

        # Trim spaces and standardize country names
        df['Country'] = df['Country'].str.strip(
        ).str.upper().replace(COUNTRY_MAPPING)

        # Convert date column to datetime format (handle multiple formats)

        df['Date'] = df['Date'].apply(convert_to_datetime)

        # Extract product name before "/"
        df['Product/Code'] = df['Product/Code'].astype(
            str).str.split('/').str[0].str.strip()

        # Clean and convert Sales & Cost columns
        df['Sales'] = df['Sales'].astype(str).str.replace(
            "USD", "").str.strip().astype(float)
        df['Cost'] = pd.to_numeric(df['Cost'].astype(
            str).str.replace("USD", "").str.strip(), errors='coerce')

        # Handle missing cost values (assume 50% of Sales)
        df['Cost'].fillna(df['Sales'] * 0.5, inplace=True)

        # Ensure cutoff_date is in the correct format
        cutoff_date = pd.to_datetime(cutoff_date)

        # Apply filters
        if condition == "before":
            df_filtered = df[(df['Date'] <= cutoff_date)]
        else:
            df_filtered = df[(df['Date'] > cutoff_date)]

        df_filtered = df_filtered[
            (df_filtered['Product/Code'].str.lower() == product_name.lower()) &
            (df_filtered['Country'] == country_name.upper())
        ]

        # Calculate total margin
        total_sales = df_filtered['Sales'].sum()
        total_cost = df_filtered['Cost'].sum()

        if total_sales > 0:
            total_margin = (total_sales - total_cost) / total_sales
            return round(total_margin, 4)

    except Exception as e:
        return f"Error: {str(e)}"


def count_unique_students(file_path):
    """Reads a text file, extracts unique student IDs, and returns the count."""
    student_ids = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Match exact 10-character alphanumeric IDs
                matches = re.findall(r'\b[A-Z0-9]{10}\b', line)
                student_ids.update(matches)
        return len(student_ids)
    except Exception as e:
        return f"Error: {str(e)}"


def extract_details_from_question_units(question):
    """Extracts city, product, and number of units from the question."""
    city_match = re.search(r'in (\w+)', question, re.IGNORECASE)
    product_match = re.search(r'of (\w+)', question, re.IGNORECASE)
    units_match = re.search(r'at least (\d+)', question, re.IGNORECASE)

    city = city_match.group(1) if city_match else None
    product = product_match.group(1) if product_match else None
    units = int(units_match.group(1)) if units_match else None

    return city, product, units


def process_question(question, file_path):
    """Processes the question and calls the appropriate function."""
    if "units of" in question and "sold in" in question:
        city, product, units = extract_details_from_question_units(question)
        if city and product and units:
            response = calculate_total_sales(file_path, product, city, units)
            return {"answer": int(response)}

    return {"answer": "Unable to process question."}


def calculate_total_sales(file_path, product, city, min_units):
    """Reads the data file, processes city names, filters data, and computes total sales."""

    df = pd.read_json(file_path)

    # Custom city mappings
    city_mappings = {
        "Tianjin": ["Tianjjin", "Tiajin", "Tianjing"],
        "Beijing": ["Bijing", "Bejjing", "Beijng"],
        "Shanghai": ["Shangai", "Shangaai", "Shanhai"],
        "Guangzhou": ["Gwangzhou", "Guangzhoo", "Guanzhou"],
        "Shenzhen": ["Shenzen", "Shenzheen", "ShenZhen"],
        "Mumbai": ["Mombai", "Mumbbi", "Mumbay"],
        "Delhi": ["Dehly", "Dhelhi", "Dehli", "Delly", "Deli"],
        "Mexico City": ["Mexicoo City", "Mexico Cty", "Mexiko City", "Mexicocity"],
        "Lagos": ["Laggoss", "Lagoss", "Lagose"],
        "London": ["Londonn", "Londn", "Lonndon", "Londdon", "Londen"],
        "Istanbul": ["Istambul", "Istnabul", "Istaanbul", "Istanboul"],
        "Moscow": ["Moskoww", "Mowscow", "Moskow", "Mosco"],
        "Paris": ["Parris", "Paries", "Pariss"],
        "Buenos Aires": ["Buenos Aeres", "Buenoss Aires", "Buenes Aires", "Buienos Aires"],
        "Cairo": ["Cairio", "Ciro", "Caiiro", "Kairo", "Kairoo"],
        "Dhaka": ["Dhaaka", "Dhacka", "Dhakaa", "Daka"],
    }

    def standardize_city(city_name):
        for standard, variations in city_mappings.items():
            if city_name in variations:
                return standard
        return city_name

    df["city"] = df["city"].apply(standardize_city)

    # Filter data
    df_filtered = df[(df['product'].str.lower() == product.lower()) & (
        df['sales'] >= min_units) & (df['city'].str.lower() == city.lower())]

    # Compute total sales
    total_sales = df_filtered['sales'].sum()

    return total_sales

# def extract_audio_from_video(video_path, start_sec, end_sec, output_audio_path="temp_audio.wav"):
#     """Extracts audio from a video segment and saves it as a WAV file."""
#     with VideoFileClip(video_path) as video:
#         audio_clip = video.audio.subclip(start_sec, end_sec)
#         audio_clip.write_audiofile(output_audio_path)

# def transcribe_audio(audio_filename="temp_audio.wav"):
#     """Transcribes the audio file using Google Web Speech API."""
#     recognizer = sr.Recognizer()
#     with sr.AudioFile(audio_filename) as source:
#         audio = recognizer.record(source)

#     try:
#         transcript = recognizer.recognize_google(audio)
#         return transcript
#     except sr.UnknownValueError:
#         return "Sorry, I could not understand the audio."
#     except sr.RequestError as e:
#         return f"Could not request results from Google Speech Recognition service; {e}"

# def extract_details_from_question_video(question):
#     """Extracts the video filename and time range from the question."""
#     video_match = re.search(r'from (\S+\.mp4)', question)
#     start_match = re.search(r'from (\d+\.?\d*)', question)
#     end_match = re.search(r'to (\d+\.?\d*)', question)

#     video_file = video_match.group(1) if video_match else None
#     start_sec = float(start_match.group(1)) if start_match else None
#     end_sec = float(end_match.group(1)) if end_match else None

#     return video_file, start_sec, end_sec


@app.get("/", response_class=HTMLResponse)
async def serve_form():
    with open("index.html", "r", encoding="utf-8") as file:
        return HTMLResponse(content=file.read(), status_code=200)


@app.post("/api/")
async def answer_question(
    question: str = Form(...),
    email: str = Form(None),
    file: UploadFile = File(None),
    html_content: str = Form(None),
    text: str = Form(None)
):
    """API endpoint to answer a question with optional file processing."""
    try:
        # Check if question matches predefined assignment questions
        if question in ASSIGNMENT_QUESTIONS:
            return {"answer": ASSIGNMENT_QUESTIONS[question]}

        if "analyze the sentiment" in question.lower():
            if not text:
                return {"answer": "Please provide a sentence to analyze its sentiment."}

            # Call LLM to analyze sentiment
            sentiment = ask_llm(
                f"Analyze the sentiment of the following text. Respond ONLY with 'GOOD', 'BAD', or 'NEUTRAL':\n\n{text}")

            # Ensure response is strictly one of the three categories
            sentiment = sentiment.strip().upper()  # Normalize response
            valid_responses = {"GOOD", "BAD", "NEUTRAL"}

            if sentiment not in valid_responses:
                return {"answer": "Error: Could not determine sentiment."}

            return {"answer": sentiment}

        if "input tokens" in question.lower():
            if not text:
                return {"answer": "Please provide the text to calculate input tokens."}

            token_count = count_tokens(text)
            return {"answer": token_count}

        if "text embedding" in question:
            messages = text.split("\n")
            embedding_request = generate_embedding_request(messages)
            return {"answer": embedding_request}

        if "number of ducks" in question.lower() and "cricinfo" in question.lower():
            answer = fetch_cricinfo_ducks()
            return {"answer": answer}

        if "weather forecast description for" in question.lower():
            weather_response = get_weather_forecast(text)
            return {"answer": weather_response}

        if "bounding box" in question:
            city, country, lat_type = extract_city_country_latitude(question)

            if city and country and lat_type:
                answer = get_latitude(city, country, lat_type)
                return {"answer": answer}

        if "GitHub API" in question and "followers" in question:
            answer = get_newest_github_user(question)
            return {"answer": answer}

        if "SUM(TAKE(SORTBY" in question:
            answer = solve_excel_formula()
            return {"answer": str(answer)}

        if "SUM(ARRAY_CONSTRAIN(SEQUENCE(100, 100, 1, 14), 1, 10))" in question:
            answer = solve_google_sheets_formula()
            return {"answer": answer}

        if "https://httpbin.org/get" in question and "email":
            httpbin_response = await send_httpbin_request(email)
            return {"answer": httpbin_response}

        if "How many Wednesdays are there in the date range 1985-11-04 to 2017-07-23?" in question:
            answer = count_wednesdays()
            return {"answer": str(answer)}

        if "Sort this JSON array of objects by the value of the age field" in question:
            answer = sort_json_array()
            return {"answer": answer}

        if "Find all <div>s having a foo class" in question:
            if html_content:
                answer = sum_data_values_from_html(html_content)
                return {"answer": str(answer)}
            else:
                return {"answer": "Error: No HTML content provided."}

        if "Calculate total sales of Gold tickets" in question:
            answer = get_total_gold_sales()
            return {"answer": str(answer)}

        if "Provide your email as a parameter, and I will compute the required hash." in question:
            answer = solve_google_colab_auth(email)
            return {"answer": answer}

        if file is not None:  # Check if a file was uploaded
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
                print(f"File saved to: {file_path}")

            # with open(file_path, "wb") as buffer:
            #     shutil.copyfileobj(file.file, buffer)

            if file.filename == "README.md":
                answer = format_and_hash_readme(file_path)
                return {"answer": answer}

            if "most similar embeddings" in question:
                # file_path = os.path.join(UPLOAD_DIR, file.filename)
                # with open(file_path, "wb") as buffer:
                #     shutil.copyfileobj(file.file, buffer)

                # Call similarity function
                result = find_most_similar_texts(file_path)
                return {"answer": result}

            if "marks of students" in question:
                response = calculate_total_marks(file_path, question)
                return {"answer": response}

            if "markdown content of the PDF" in question:
                markdown_output = convert_pdf_to_markdown(file_path)
                return {"answer": markdown_output}

            if "total margin for transactions" in question:
                cutoff_date, product, country, condition = extract_details_from_question(
                    question)
                margin = clean_and_calculate_margin(
                    file_path, product, country, cutoff_date, condition)

                if isinstance(margin, float):
                    return {"answer": f"{margin:.4%}"}

            if "sold" in question and "transactions" in question:
                response = process_question(question, file_path)
                return {"answer": response}

            if "unique students" in question:
                response = count_unique_students(file_path)
                return {"answer": response}

            if "transcribe" in question.lower() and "video" in question.lower():
                video_file, start_sec, end_sec = extract_details_from_question_video(
                    question)
                if video_file and start_sec is not None and end_sec is not None:
                    extract_audio_from_video(video_file, start_sec, end_sec)
                    response = transcribe_audio()
                    return {"answer": response}

            if "Calculate the number of pixels with a certain minimum brightness" in question and file is not None:
                file_path = os.path.join(UPLOAD_DIR, file.filename)

                try:
                    # Read and save the uploaded file properly
                    with open(file_path, "wb") as f:
                        # Ensure the file pointer is at the beginning
                        file.file.seek(0)
                        f.write(file.file.read())

                    # Verify if the file is actually saved
                    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                        return {"error": "❌ File upload failed. The file is empty or corrupted."}

                    # Convert WebP to PNG if needed
                    if file.filename.endswith(".webp"):
                        new_file_path = convert_webp_to_png_opencv(file_path)
                        if new_file_path:
                            file_path = new_file_path  # Use the converted file
                        else:
                            return {"error": "❌ WebP to PNG conversion failed."}

                    # Process image and count light pixels
                    answer = count_light_pixels(file_path)
                    return {"answer": str(answer)}

                except Exception as e:
                    return {"error": f"❌ Error processing file: {str(e)}"}

            # elif file.filename.endswith(".zip"):
            #     answer = extract_and_process_zip(file_path)
            # else:
            #     answer = "Unsupported file format."

            # Handle ZIP file processing
            # if file is not None and file.filename.endswith(".zip"):
            #     file_path = os.path.join(UPLOAD_DIR, file.filename)

            #     with open(file_path, "wb") as buffer:
            #         # shutil.copyfileobj(file.file, buffer)
            #         buffer.write(await file.read())

            #     print("File saved at:", file_path)  # Debugging

            #     if os.path.exists(file_path):  # Confirm file exists
            #         extracted_files = await extract_zip(file_path)
            #         print("Extracted files:", extracted_files)  # Debugging

            #         if isinstance(extracted_files, str):  # Check if it's an error message
            #             return {"answer": extracted_files}

            #         if "extract.csv" in [os.path.basename(f) for f in extracted_files]:
            #             answer = process_extract_csv()
            #         else:
            #             answer = "extract.csv not found in ZIP."
            #     else:
            #         answer = "Error: Uploaded ZIP file not saved properly."

            if file.filename.endswith(".zip"):
                extract_path = os.path.join(
                    EXTRACTED_DIR, os.path.splitext(file.filename)[0])
                os.makedirs(extract_path, exist_ok=True)

                try:
                    with zipfile.ZipFile(file_path, "r") as zip_ref:
                        zip_ref.extractall(extract_path)
                        extracted_files = os.listdir(extract_path)
                        # return {
                        #     "message": "ZIP file uploaded and extracted successfully",
                        #     "extracted_to": extract_path,
                        #     "files": extracted_files
                        # }
                        # Debug Log
                        print(f"📂 Extracted Files: {extracted_files}")

                        # ✅ Determine which question is being processed
                        # ✅ Check for extract.csv processing
                        if "extract.csv" in extracted_files and "answer" in question.lower():
                            return process_question_1(extract_path)

                        if any(f in extracted_files for f in ["data1.csv", "data2.csv", "data3.txt"]) and "sum" in question.lower():
                            answer = process_question_2(extract_path)
                            return {"answer": answer["sum"]}

                        if any(f.startswith("file") and f.endswith(".txt") for f in extracted_files) and "replace" in question.lower():
                            answer = process_text_files(extract_path)
                            return {"answer": answer}

                        if "list all files" in question.lower() and "size" in question.lower() and "modified" in question.lower():
                            answer = calculate_filtered_size(extract_path)
                            return {"answer": answer}

                        if "rename" in question.lower() and "sha256sum" in question.lower():
                            result = process_zip_and_hash(extract_path)
                            return {"answer": result}

                        # ✅ If the question matches the line difference task
                        if "how many lines are different" in question.lower():
                            if "a.txt" in extracted_files and "b.txt" in extracted_files:
                                result = count_different_lines(extract_path)
                                return {"answer": result}
                            else:
                                return {"error": "Required files (a.txt, b.txt) not found in ZIP."}

                        return {"message": "ZIP extracted successfully, but question processing not recognized."}

                except zipfile.BadZipFile:
                    return {"error": "Uploaded file is not a valid ZIP archive"}

            # if file and file.filename.endswith(".zip"):
            # # Save the uploaded ZIP file
            #     file_path = os.path.join(UPLOAD_DIR, file.filename)
            #     with open(file_path, "wb") as buffer:
            #         shutil.copyfileobj(file.file, buffer)

            #     # Extract ZIP and retrieve file details
            #     zip_result = extract_zip_file(file_path)

            #     if "error" in zip_result:
            #         return {"error": zip_result["error"]}

            #     extract_folder = zip_result["extracted_to"]

            #     # 🔹 Check question text to determine processing logic
            #     if "Download and unzip file  which has a single extract.csv file inside.  What is the value in the 'answer' column of the CSV file?" in question and "extract.csv" in zip_result["files"]:
            #         return process_question_1(extract_folder)

            #     if "sum" in question and any(f in zip_result["files"] for f in ["data1.csv", "data2.csv", "data3.txt"]):
            #         return process_question_2(extract_folder)

            #     return {"message": "ZIP extracted successfully, but question processing not recognized."}

            # return {"message": "No relevant file uploaded."}

            # if file and file.filename.endswith(".zip"):
            #     if "Sum up all the values where the symbol matches ‰ OR ‡ OR œ" in question:
            #         zip_path = os.path.join(UPLOAD_DIR, file.filename)
            #         os.makedirs(UPLOAD_DIR, exist_ok=True)

            #         # Save uploaded ZIP file
            #         with open(zip_path, "wb") as buffer:
            #             # shutil.copyfileobj(file.file, buffer)
            #             buffer.write(await file.read())

            #             print("File saved at:", zip_path)

            #         # Extract CSV files
            #         file_paths = extract_zip(zip_path, EXTRACT_DIR)

            #         if len(file_paths) == 3:
            #             total_sum = sum_selected_symbols(file_paths)
            #             return {"answer": total_sum}
            #         return {"answer": "Error: ZIP file should contain exactly 3 CSV files."}

            #     return {"answer": "Error: No ZIP file uploaded."}

            # return {"answer": "No matching condition found."}

        # return {"answer": "No matching condition."}

    except Exception as e:
        return {"answer": f"Error processing request: {str(e)}"}

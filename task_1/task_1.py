import os
import json
import pandas as pd
import pdfplumber
import fitz  # PyMuPDF

# Paths
INPUT_DIR = "pdfs"
OUTPUT_DIR = "output"
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
KV_DIR = os.path.join(OUTPUT_DIR, "key_values")
TEXT_DIR = os.path.join(OUTPUT_DIR, "text")

# Create output directories
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(KV_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)


def extract_key_values_and_text(pdf_path):
    doc = fitz.open(pdf_path)
    key_values = {}
    full_text = ""
    for page in doc:
        text = page.get_text()
        full_text += text + "\n"
        lines = text.split("\n")
        for line in lines:
            if ":" in line:
                parts = line.split(":", 1)
                key = parts[0].strip()
                value = parts[1].strip()
                if key and value:
                    key_values[key] = value
    return key_values, full_text


def extract_tables(pdf_path):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_tables = page.extract_tables()
            for table in extracted_tables:
                df = pd.DataFrame(table[1:], columns=table[0])
                tables.append(df)
    return tables


def save_output(pdf_name, key_values, tables, full_text):
    base_name = os.path.splitext(pdf_name)[0]

    # Save key-values as JSON
    with open(os.path.join(KV_DIR, f"{base_name}_key_values.json"), "w", encoding="utf-8") as f:
        json.dump(key_values, f, indent=4)

    # Save extracted text as .txt
    with open(os.path.join(TEXT_DIR, f"{base_name}_text.txt"), "w", encoding="utf-8") as f:
        f.write(full_text)

    # Save each table to CSV
    for i, table_df in enumerate(tables):
        csv_path = os.path.join(TABLE_DIR, f"{base_name}_table_{i+1}.csv")
        table_df.to_csv(csv_path, index=False)


def process_all_pdfs():
    pdf_files = [f for f in os.listdir(
        INPUT_DIR) if f.lower().endswith(".pdf")]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(INPUT_DIR, pdf_file)
        print(f"Processing: {pdf_file}")

        key_values, full_text = extract_key_values_and_text(pdf_path)
        tables = extract_tables(pdf_path)

        save_output(pdf_file, key_values, tables, full_text)

        print(f"Finished: {pdf_file}\n")


if __name__ == "__main__":
    process_all_pdfs()

import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict

# --- Configuration ---
FILE_PATH_1 = 'data/raw/pets.md' 
FILE_PATH_2 = 'data/raw/travel.md' 
OUTPUT_FILE = 'data/processed/deduplicated_output.md' 
MODEL_NAME = 'all-MiniLM-L6-v2' # embedding model
SIMILARITY_THRESHOLD = 0.90 # threshold fr similar docs
CHUNK_HEADER_REGEX = r'\n(#{2,3}\s.*)' # regex to find H2 and H3 headers

# read file and return content
def read_file(fp): 
    try:
        with open(fp, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {fp}")
        return ""
    except Exception as e:
        print(f"Error reading file {fp}: {e}")
        return ""

# chunks MD text based on H2/H3 headers
def chunk_markdown(text: str, file_origin: str) -> List[Dict]:
    raw_splits = re.split(CHUNK_HEADER_REGEX, text)
    chunks = []
    current_header = "Introduction" # for content before h2/h3

    # for initial h1
    if raw_splits[0] and raw_splits[0].strip():
         chunks.append({
             "header": current_header,
             "content": raw_splits[0].strip(),
             "file": file_origin,
             "original_index": len(chunks)
         })

    # pairs of (header, content)
    for i in range(1, len(raw_splits), 2):
        header = raw_splits[i].strip()
        content = raw_splits[i+1].strip() if (i+1) < len(raw_splits) else ""
        if header: current_header = header
        if content:
            chunks.append({
                "header": current_header,
                "content": content,
                "file": file_origin,
                "original_index": len(chunks)
            })
    print(f"Chunked {file_origin}: Found {len(chunks)} chunks.")
    return chunks


content1 = read_file(FILE_PATH_1)
content2 = read_file(FILE_PATH_2)

if not content1 and not content2:
    print("Both input files are empty or could not be read.")
    exit()

chunks1 = chunk_markdown(content1, FILE_PATH_1)
chunks2 = chunk_markdown(content2, FILE_PATH_2)
all_chunks = chunks1 + chunks2

if not all_chunks:
    print("No content chunks found in the files.")
    exit()

for i, chunk in enumerate(all_chunks):
    chunk['id'] = i # unique id for chunks

print(f"Loading sentence transformer model: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)

print("Generating embeddings for all chunks...")
chunk_contents = [chunk['content'] for chunk in all_chunks]
embeddings = model.encode(chunk_contents, show_progress_bar=True)
print(f"Generated {len(embeddings)} embeddings.")

print("Calculating cosine similarity matrix...")
similarity_matrix = cosine_similarity(embeddings)
print(f"Similarity matrix shape: {similarity_matrix.shape}")

# Identify Duplicates
num_chunks = len(all_chunks)
indices_to_remove = set()

print(f"Identifying duplicates with threshold > {SIMILARITY_THRESHOLD}...")
# Iterate through the upper triangle of the similarity matrix
for i in range(num_chunks):
    if i in indices_to_remove:
        continue
    for j in range(i + 1, num_chunks):
        if j in indices_to_remove:
            continue

        similarity = similarity_matrix[i, j]

        if similarity > SIMILARITY_THRESHOLD:
            # Found a pair of similar chunks
            chunk_i = all_chunks[i]
            chunk_j = all_chunks[j]

            # --- Deduplication Rule: Keep the chunk with more content (longer) ---
            len_i = len(chunk_i['content'])
            len_j = len(chunk_j['content'])

            print(f"  - Found similar pair (Sim: {similarity:.4f}):")
            print(f"    Chunk {i} (File: {chunk_i['file']}, Header: '{chunk_i['header'][:30]}...', Length: {len_i})")
            print(f"    Chunk {j} (File: {chunk_j['file']}, Header: '{chunk_j['header'][:30]}...', Length: {len_j})")

            if len_i >= len_j:
                indices_to_remove.add(j)
                print(f"    -> Marking shorter chunk {j} for removal.")
            else:
                indices_to_remove.add(i)
                print(f"    -> Marking shorter chunk {i} for removal.")
                break 

final_chunks = []
for i in range(num_chunks):
    if i not in indices_to_remove:
        final_chunks.append(all_chunks[i])

print(f"Original total chunks: {num_chunks}")
print(f"Final unique chunks: {len(final_chunks)}")
print(f"Combining unique chunks into {OUTPUT_FILE}...")
output_content = ""
final_chunks_sorted = sorted(final_chunks, key=lambda x: (x['file'], x['original_index']))

is_first_chunk = True
for chunk in final_chunks_sorted:
    if not is_first_chunk:
         output_content += "\n\n"
    
    if chunk['header'] != "Introduction": 
         output_content += f"{chunk['header']}\n" 
    output_content += chunk['content']
    is_first_chunk = False

try:
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(output_content)
    print(f"Successfully saved deduplicated content to {OUTPUT_FILE}")
except Exception as e:
    print(f"Error writing output file: {e}")
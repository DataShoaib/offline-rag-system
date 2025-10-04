import os
import chromadb
from cryptography.fernet import Fernet
import logging

logging.basicConfig(filename="logs/cleanup.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load encryption key
ENCRYPTION_KEY_PATH = "encryption.key"
with open(ENCRYPTION_KEY_PATH, "rb") as key_file:
    key = key_file.read()
fernet = Fernet(key)

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("rag_collection")

# Get all documents
documents = collection.get(include=["metadatas"])

# Check each document
invalid_ids = []
for i, meta in enumerate(documents["metadatas"]):
    file_id = meta.get("file_id")
    if not file_id:
        invalid_ids.append(documents["ids"][i])
        logging.warning(f"Document {documents['ids'][i]} has no file_id. Marking for deletion.")
        continue
    enc_path = f"encrypted_files/{file_id}"
    meta_path = f"encrypted_files/{file_id}.meta"
    if not os.path.exists(enc_path):
        invalid_ids.append(documents["ids"][i])
        logging.warning(f"Encrypted file {file_id} missing. Marking document {documents['ids'][i]} for deletion.")
        continue
    if not os.path.exists(meta_path):
        invalid_ids.append(documents["ids"][i])
        logging.warning(f"Metadata file {file_id}.meta missing. Marking document {documents['ids'][i]} for deletion.")
        continue
    try:
        with open(enc_path, "rb") as ef:
            enc_bytes = ef.read()
        if not enc_bytes:
            invalid_ids.append(documents["ids"][i])
            logging.warning(f"Encrypted file {file_id} is empty. Marking document {documents['ids'][i]} for deletion.")
            continue
        fernet.decrypt(enc_bytes[:100])  # Test decryption
    except Exception as e:
        invalid_ids.append(documents["ids"][i])
        logging.warning(f"File {file_id} failed decryption: {str(e)}. Marking document {documents['ids'][i]} for deletion.")

# Delete invalid documents
if invalid_ids:
    collection.delete(ids=invalid_ids)
    logging.info(f"Deleted {len(invalid_ids)} invalid documents from vector store: {invalid_ids}")
else:
    logging.info("No invalid documents found in vector store.")

print(f"Cleanup complete. Check logs/cleanup.log for details.")
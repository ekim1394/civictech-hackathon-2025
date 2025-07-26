# Populate a vector index with embeddings from Amazon Titan Text Embeddings V2.
import boto3
import json
from bs4 import BeautifulSoup
from pathlib import Path
import PyPDF2
import dotenv

dotenv.load_dotenv()

# Create Bedrock Runtime and S3 Vectors clients in the AWS Region of your choice. 
bedrock = boto3.client("bedrock-runtime", region_name='us-east-1')
s3vectors = boto3.client("s3vectors", region_name='us-east-1')

def extract_text_from_html(html_content):
    """Extract text from HTML content, removing HTML tags."""
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text(separator=' ', strip=True)

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + " "
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        return ""

def get_embedding(text):
    """Generate embedding for a given text using Amazon Titan Text Embeddings V2."""
    # Truncate text if it's too long (Titan has a context limit)
    if len(text) > 8000:
        text = text[:8000]
        
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({"inputText": text})
    )

    # Extract embedding from response.
    response_body = json.loads(response["body"].read())
    return response_body["embedding"]

# Process CMS documents
cms_docs = []
base_path = Path("raw_data/CMS")
cms_dirs = [d for d in base_path.iterdir() if d.is_dir()]

# Process HTML documents from text directories
for cms_dir in cms_dirs:
    # Navigate to the text directory
    text_dir = next((d for d in cms_dir.iterdir() if d.name.startswith("text-")), None)
    if not text_dir:
        continue
    
    # Navigate to the documents directory
    docs_dir = text_dir / "documents"
    if not docs_dir.exists():
        continue
    
    # Process each document
    for file_path in docs_dir.iterdir():
        if file_path.name.endswith("_content.htm"):
            # Extract document ID from filename
            doc_id = file_path.name.split("_")[0]
            
            # Read HTML content
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Extract text from HTML
            text_content = extract_text_from_html(html_content)
            
            # Get metadata from JSON file if available
            json_path = docs_dir / f"{doc_id}.json"
            metadata = {}
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    # Extract useful metadata
                    attributes = json_data.get('data', {}).get('attributes', {})
                    metadata = {
                        "title": attributes.get("title", ""),
                        "document_type": attributes.get("documentType", ""),
                        "agency_id": attributes.get("agencyId", ""),
                        "docket_id": attributes.get("docketId", ""),
                        "posted_date": attributes.get("postedDate", "")
                    }
            
            # Add to our collection
            cms_docs.append({
                "id": doc_id,
                "text": text_content,
                "metadata": metadata,
                "source_type": "html"
            })

# Process PDF files from binary directories
for cms_dir in cms_dirs:
    # Navigate to the binary directory
    binary_dir = next((d for d in cms_dir.iterdir() if d.name.startswith("binary-")), None)
    if not binary_dir:
        continue
    
    # Check for comments_attachments directory
    attachments_dir = binary_dir / "comments_attachments"
    if not attachments_dir.exists():
        continue
    
    # Process each PDF file
    for file_path in attachments_dir.iterdir():
        if file_path.name.endswith(".pdf"):
            # Extract document ID and attachment number from filename
            # Format: CMS-2025-0013-0002_attachment_1.pdf
            parts = file_path.stem.split("_")
            doc_id = parts[0]
            attachment_num = parts[2] if len(parts) > 2 else "1"
            
            # Extract text from PDF
            text_content = extract_text_from_pdf(file_path)
            
            if text_content:
                # Add to our collection
                cms_docs.append({
                    "id": f"{doc_id}_attachment_{attachment_num}",
                    "text": text_content,
                    "metadata": {
                        "document_id": doc_id,
                        "attachment_number": attachment_num,
                        "file_name": file_path.name
                    },
                    "source_type": "pdf"
                })

print(f"Found {len(cms_docs)} documents")

# Generate embeddings for each document
print("Generating embeddings...")
for i, doc in enumerate(cms_docs):
    print(f"Processing document {i+1}/{len(cms_docs)}: {doc['id']}")
    doc["embedding"] = get_embedding(doc["text"])

# Write embeddings into vector index with metadata
print("Writing embeddings to vector index...")
vectors = []
for doc in cms_docs:
    # Create metadata dictionary based on source type
    if doc["source_type"] == "html":
        metadata = {
            "source_text": doc["text"][:1000],  # Truncate text for metadata
            "title": doc["metadata"].get("title", ""),
            "document_type": doc["metadata"].get("document_type", ""),
            "agency_id": doc["metadata"].get("agency_id", ""),
            "docket_id": doc["metadata"].get("docket_id", ""),
            "posted_date": doc["metadata"].get("posted_date", ""),
            "source_type": "html"
        }
    else:  # pdf
        metadata = {
            "source_text": doc["text"][:1000],  # Truncate text for metadata
            "document_id": doc["metadata"].get("document_id", ""),
            "attachment_number": doc["metadata"].get("attachment_number", ""),
            "file_name": doc["metadata"].get("file_name", ""),
            "source_type": "pdf"
        }
    
    vectors.append({
        "key": doc["id"],
        "data": {"float32": doc["embedding"]},
        "metadata": metadata
    })

# Write embeddings to a JSON file
# with open('cms_embeddings.json', 'w', encoding='utf-8') as f:
#     json.dump(vectors, f, indent=4, ensure_ascii=False)

# print(f"Successfully wrote {len(vectors)} document embeddings to cms_embeddings.json")

# Put vectors in batches if there are many
if vectors:
    s3vectors.put_vectors(
        vectorBucketName="ekim-civictech-hackathon-2025",   
        indexName="cms-titan",   
        vectors=vectors
    )
    print(f"Successfully added {len(vectors)} document embeddings to vector index")
else:
    print("No documents found to process")
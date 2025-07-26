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

# Process docket summary files
for cms_dir in cms_dirs:
    # Navigate to the text directory
    text_dir = next((d for d in cms_dir.iterdir() if d.name.startswith("text-")), None)
    if not text_dir:
        continue
    
    # Navigate to the docket directory
    docket_dir = text_dir / "docket"
    if not docket_dir.exists():
        continue
    
    # Process docket summary file (typically named after the docket ID)
    for file_path in docket_dir.iterdir():
        if file_path.name.endswith(".json"):
            # Read JSON content
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    json_data = json.load(f)
                    
                    # Extract docket ID
                    docket_id = json_data.get('data', {}).get('id', '')
                    
                    # Extract docket attributes
                    attributes = json_data.get('data', {}).get('attributes', {})
                    
                    # Extract docket abstract/description
                    docket_abstract = attributes.get('dkAbstract', '')
                    
                    
                    # Add to our collection
                    cms_docs.append({
                        "id": f"{docket_id}_docket_summary",
                        "text": docket_abstract if docket_abstract else "No abstract available",
                        "source_type": "docket_summary"
                    })
                    
                except Exception as e:
                    print(f"Error processing docket file {file_path}: {e}")

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
                "id": doc_id + "_htm",
                "text": text_content if text_content else "No text available",
                "metadata": metadata,
                "source_type": "html"
            })

# Process comments from text directories
for cms_dir in cms_dirs:
    # Navigate to the text directory
    text_dir = next((d for d in cms_dir.iterdir() if d.name.startswith("text-")), None)
    if not text_dir:
        continue
    
    # Navigate to the comments directory
    comments_dir = text_dir / "comments"
    if not comments_dir.exists():
        continue
    
    # Process each comment JSON file
    for file_path in comments_dir.iterdir():
        if file_path.name.endswith(".json"):
            # Read JSON content
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    json_data = json.load(f)
                    
                    # Extract comment ID
                    comment_id = json_data.get('data', {}).get('id', '')
                    
                    # Extract comment text
                    attributes = json_data.get('data', {}).get('attributes', {})
                    comment_text = attributes.get('comment', '')
                    
                    # Extract metadata
                    metadata = {
                        "title": attributes.get("title", ""),
                        "organization": attributes.get("organization", ""),
                        "document_type": attributes.get("documentType", ""),
                        "agency_id": attributes.get("agencyId", ""),
                        "docket_id": attributes.get("docketId", ""),
                        "posted_date": attributes.get("postedDate", ""),
                        "comment_on_document_id": attributes.get("commentOnDocumentId", "")
                    }
                    
                    # Check for attachments
                    has_attachments = False
                    if json_data.get('data', {}).get('relationships', {}).get('attachments', {}).get('data'):
                        has_attachments = True
                    
                    # If comment text is empty or just refers to attachments, note this in metadata
                    if not comment_text or comment_text.lower() == "see attached file(s)":
                        metadata["comment_in_attachments"] = True
                    
                    # Add to our collection
                    cms_docs.append({
                        "id": comment_id + "_comment",
                        "text": comment_text if comment_text else "No comment text available",
                        "metadata": metadata,
                        "source_type": "comment",
                        "has_attachments": has_attachments
                    })
                    
                except Exception as e:
                    print(f"Error processing comment file {file_path}: {e}")

# # Process PDF files from binary directories
# for cms_dir in cms_dirs:
#     # Navigate to the binary directory
#     binary_dir = next((d for d in cms_dir.iterdir() if d.name.startswith("binary-")), None)
#     if not binary_dir:
#         continue
    
#     # Check for comments_attachments directory
#     attachments_dir = binary_dir / "comments_attachments"
#     if not attachments_dir.exists():
#         continue
    
    # Process each PDF file
    # for file_path in attachments_dir.iterdir():
    #     if file_path.name.endswith(".pdf"):
    #         continue
            # # Extract document ID and attachment number from filename
            # # Format: CMS-2025-0013-0002_attachment_1.pdf
            # parts = file_path.stem.split("_")
            # doc_id = parts[0]
            # attachment_num = parts[2] if len(parts) > 2 else "1"
            
            # # Extract text from PDF
            # text_content = extract_text_from_pdf(file_path)
            
            # if text_content:
            #     # Add to our collection
            #     cms_docs.append({
            #         "id": f"{doc_id}_attachment_{attachment_num}",
            #         "text": text_content,
            #         "metadata": {
            #             "document_id": doc_id,
            #             "attachment_number": attachment_num,
            #             "file_name": file_path.name
            #         },
            #         "source_type": "pdf"
            #     })

print(f"Found {len(cms_docs)} documents")

# Generate embeddings for each document
print("Generating embeddings...")
for i, doc in enumerate(cms_docs):
    print(f"Processing document {i+1}/{len(cms_docs)}: {doc['id']}")
    doc["embedding"] = get_embedding(doc['id'] + " " + doc["text"])

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
    elif doc["source_type"] == "comment":
        metadata = {
            "source_text": doc["text"][:1000],  # Truncate text for metadata
            "title": doc["metadata"].get("title", ""),
            "document_type": doc["metadata"].get("document_type", ""),
            "agency_id": doc["metadata"].get("agency_id", ""),
            "docket_id": doc["metadata"].get("docket_id", ""),
            "posted_date": doc["metadata"].get("posted_date", ""),
            "comment_on_document_id": doc["metadata"].get("comment_on_document_id", ""),
            "source_type": "comment"
        }
    elif doc["source_type"] == "docket_summary":
        metadata = {
            "source_text": doc["text"][:1000],  # Truncate text for metadata
            "title": doc["metadata"].get("title", ""),
            "agency_id": doc["metadata"].get("agency_id", ""),
            "docket_type": doc["metadata"].get("docket_type", ""),
            "effective_date": doc["metadata"].get("effective_date", ""),
            "modify_date": doc["metadata"].get("modify_date", ""),
            "rin": doc["metadata"].get("rin", ""),
            "source_type": "docket_summary"
        }
    else:  # pdf
        continue
        # metadata = {
        #     "source_text": doc["text"][:1000],  # Truncate text for metadata
        #     "document_id": doc["metadata"].get("document_id", ""),
        #     "attachment_number": doc["metadata"].get("attachment_number", ""),
        #     "file_name": doc["metadata"].get("file_name", ""),
        #     "source_type": "pdf"
        # }
    
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
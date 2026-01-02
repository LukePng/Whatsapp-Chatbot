import os
import base64
import uuid
from PIL import Image
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

load_dotenv()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
FIGURES_PATH = os.path.join(BASE_DIR, "figures")
POPPLER_PATH = r"C:\poppler\Library\bin"

# Models
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
vision_model = ChatOpenAI(model="gpt-4o")
tagger_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def describe_physics_diagram(image_path):
    """Filters out logos and QR codes by asking GPT-4o to validate content."""
    with open(image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    response = vision_model.invoke([
        ("system", "You are a physics expert. If this is a diagram (circuit, vector, graph), describe it. "
                   "If it is a LOGO, QR CODE, or just a small snippet of text/math, reply ONLY with 'SKIP'."),
        ("user", [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}])
    ])
    return response.content

# --- Initialize Vector Store ---
vector_store = Chroma(collection_name="physics_collection", embedding_function=embeddings_model, persist_directory=CHROMA_PATH)

for pdf_name in [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]:
    print(f"--- Processing {pdf_name} ---")
    
    # Use 'yolox' to accurately find diagrams amidst the dense text
    elements = partition_pdf(
        filename=os.path.join(DATA_PATH, pdf_name),
        extract_images_in_pdf=True,
        image_output_dir_path=FIGURES_PATH,
        poppler_path=POPPLER_PATH,
        strategy="hi_res",
        hi_res_model_name="yolox",
        chunking_strategy="by_title",
        max_characters=800
    )

    current_topic = "General"
    final_docs = []

    for el in elements:
        text_content = str(el)
        
        # DYNAMIC TOPIC UPDATER: Detects when a new chapter starts
        if "Chapter" in text_content or "Topic" in text_content:
            topic_query = tagger_llm.invoke(f"Extract only the Chapter/Topic name from this line: '{text_content}'")
            current_topic = topic_query.content.strip().replace(".", "")
            print(f"   Switching Topic to: {current_topic}")

        metadata = {"filename": pdf_name, "topic": current_topic}
        
        if "CompositeElement" in str(type(el)):
            # Clean out the 'The Game Changer' repetitive text
            cleaned_text = text_content.replace("THE GAME CHANGER", "").strip()
            if len(cleaned_text) > 20:
                final_docs.append(Document(page_content=cleaned_text, metadata=metadata))
            
        elif "Image" in str(type(el)):
            img_path = el.metadata.image_path
            
            # SIZE FILTER: Deletes tiny icons or QR code fragments
            with Image.open(img_path) as img:
                w, h = img.size
            if w < 180 or h < 180:
                os.remove(img_path)
                continue
            
            # AI VALIDATION: Deletes logos and non-diagrams
            description = describe_physics_diagram(img_path)
            if "SKIP" in description.upper():
                os.remove(img_path)
                continue
            
            metadata.update({"source": "image", "image_path": img_path})
            final_docs.append(Document(page_content=f"[DIAGRAM]: {description}", metadata=metadata))

    if final_docs:
        vector_store.add_documents(documents=final_docs, ids=[str(uuid.uuid4()) for _ in final_docs])

print("Training Complete! Every chapter is correctly filed.")
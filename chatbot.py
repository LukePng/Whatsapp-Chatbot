import os
import logging
import gradio as gr
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Silence logs
logging.getLogger("httpx").setLevel(logging.WARNING)
load_dotenv()

# --- Setup Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
FIGURES_PATH = os.path.join(BASE_DIR, "figures")

# --- Initialize Models ---
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = Chroma(
    collection_name="physics_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH, 
)
llm = ChatOpenAI(temperature=0.1, model='gpt-4o-mini')

def stream_response(message, history):
    # 1. PRIORITY DISPATCHER (Weighted towards the most recent turn)
    # This prevents the bot from staying 'stuck' in Mechanics memory.
    recent_turn = str(history[-1:]) if history else "None"
    
    routing_prompt = f"""
    Identify the current A-Level Physics topic. 
    Prioritize the context of the LAST message in the history.
    
    Last Turn: {recent_turn}
    Current Question: "{message}"
    
    If the question is "derivation?", look at the topic mentioned in the Last Turn.
    Options: Mechanics, Waves, Electricity, Thermal, Quantum, Measurements, General.
    Return ONLY the single word.
    """
    detected_topic = llm.invoke(routing_prompt).content.strip().replace(".", "")

    # 2. HARD TOPIC LOCK (Strict Filter)
    # This physically prevents Electricity/Mechanics leakage when in Waves.
    search_kwargs = {"k": 6}
    if detected_topic != "General":
        search_kwargs["filter"] = {"topic": detected_topic}

    docs = vector_store.as_retriever(search_kwargs=search_kwargs).invoke(message)
    
    knowledge = ""
    relevant_image = None
    for doc in docs:
        knowledge += f"{doc.page_content}\n\n"
        if not relevant_image and doc.metadata.get("source") == "image":
            relevant_image = doc.metadata.get("image_path")

    # 3. SHORT-MEMORY TUTOR PROMPT
    # We limit history to 2 turns to keep the bot focused on the current subject.
    rag_prompt = f"""
    Role: Expert A-Level Physics Tutor.
    Topic: {detected_topic}.
    
    Instructions:
    - Use the provided Context to answer accurately. 
    - If the student changed topics (e.g. to Waves), ignore old Mechanics formulas in the history.
    - If the user asks for a derivation, look for the STEP-BY-STEP proof in the Context.
    - Use LaTeX for all math ($F=ma$) and bold for **key terms**.
    
    Context: {knowledge}
    History: {history[-2:] if history else "Start of session"}
    Question: {message}
    """

    partial_message = ""
    for response in llm.stream(rag_prompt):
        partial_message += response.content
        yield partial_message

    if relevant_image:
        yield partial_message + f"\n\n![Related Diagram](file/{relevant_image})"

# --- Launch UI ---
chatbot_ui = gr.ChatInterface(
    stream_response, 
    chatbot=gr.Chatbot(height=600, render_markdown=True),
    title="A-Level Physics Tutor"
)

if __name__ == "__main__":
    chatbot_ui.launch(share=True, allowed_paths=[FIGURES_PATH])
import pandas as pd
import json
import os
import re
import pickle
from typing import List, Dict, Any
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from pydantic import BaseModel, Field

load_dotenv()

class MovieResponse(BaseModel):
    answer: str = Field(description="Natural language answer referencing specific movies")
    contexts: List[str] = Field(description="Plot snippets used to form the answer")
    reasoning: str = Field(description="Simple 1-2 sentence explanation: what did the question ask, what did you search for, what did you find, how did you use it to form the answer")

class RAGSystem:
    def __init__(self):
        # Set up the RAG system with embeddings model and LLM
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"),
            model_kwargs={'device': 'cpu'}
        )
        
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=os.getenv("LLM_MODEL", "qwen/qwen3-32b"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.6")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1000")),
            reasoning_effort="default",
            reasoning_format="parsed"
        ).with_structured_output(MovieResponse, method="json_mode")
        
        self.vector_store = None
        self.bm25 = None
        self.documents = None
        self.chain = None

    def _clean_text(self, text: str) -> str:
        # Remove Wikipedia tags and extra whitespace from text
        return re.sub(r'\s+', ' ', re.sub(r'\[edit\]', '', text)).strip()
    
    def _cache_exists(self, cache_dir: str) -> bool:
        # Check if all required cache files exist on disk
        return all(os.path.exists(os.path.join(cache_dir, f)) 
                   for f in ["index.faiss", "index.pkl", "bm25_index.pkl", "documents.pkl"])
    
    def _load_cache(self, cache_dir: str):
        # Load pre-built vector store and BM25 index from cache to skip rebuilding
        print(f"Loading cached indices from {cache_dir}...")
        self.vector_store = FAISS.load_local(cache_dir, self.embeddings, allow_dangerous_deserialization=True)
        with open(os.path.join(cache_dir, "bm25_index.pkl"), "rb") as f:
            self.bm25 = pickle.load(f)
        with open(os.path.join(cache_dir, "documents.pkl"), "rb") as f:
            self.documents = pickle.load(f)
        print("Cached indices loaded!")
    
    def _save_cache(self, cache_dir: str):
        # Save vector store and BM25 index to disk for faster future runs
        print(f"Saving indices to {cache_dir}...")
        os.makedirs(cache_dir, exist_ok=True)
        self.vector_store.save_local(cache_dir)
        with open(os.path.join(cache_dir, "bm25_index.pkl"), "wb") as f:
            pickle.dump(self.bm25, f)
        with open(os.path.join(cache_dir, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        print("Indices saved!")

    def load_and_process_data(self):
        # Load last 400 movies from CSV, create chunks, and build search indices
        cache_dir = os.getenv("VECTOR_DB_PATH", "./vector_db/")
        use_cache = os.getenv("USE_CACHE", "true").lower() == "true"
        
        if use_cache and self._cache_exists(cache_dir):
            try:
                self._load_cache(cache_dir)
                return
            except Exception as e:
                print(f"Cache load failed: {e}. Rebuilding...")
        
        csv_path = os.getenv("DATASET_PATH", "wiki_movie_plots_deduped.csv")
        print(f"Loading dataset from {csv_path}...")
        
        df = pd.read_csv(csv_path)
        df = df[df['Plot'].str.len() > 200].tail(400)
        print(f"Selected last {len(df)} movies. Year range: {df['Release Year'].min()}-{df['Release Year'].max()}")
        
        df['Plot'] = df['Plot'].apply(self._clean_text)
        df['searchable_text'] = (
            "Title: " + df['Title'].astype(str) + 
            ". Director: " + df['Director'].astype(str) + 
            ". Plot: " + df['Plot'].astype(str)
        )
        
        documents = [Document(
            page_content=row['searchable_text'],
            metadata={"title": row['Title'], "year": row['Release Year'], "genre": row['Genre'], "director": row['Director']}
        ) for _, row in df.iterrows()]
        
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", "400")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100"))
        ).split_documents(documents)
        print(f"Created {len(chunks)} chunks.")
        
        self.documents = chunks
        self.bm25 = BM25Okapi([doc.page_content.lower().split() for doc in chunks])
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        print("Vector store and BM25 index created.")
        
        if use_cache:
            self._save_cache(cache_dir)
    
    def hybrid_search(self, query: str, k: int = 3) -> List[Document]:
        # Combine keyword search (BM25) and semantic search (FAISS) for better results
        k = int(os.getenv("TOP_K_RESULTS", k))
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k*2]
        
        combined = {}
        for idx in bm25_indices:
            doc = self.documents[idx]
            doc_id = f"{doc.metadata.get('title')}_{doc.page_content[:50]}"
            combined[doc_id] = {'doc': doc, 'bm25': bm25_scores[idx], 'semantic': 0}
        
        for i, doc in enumerate(self.vector_store.similarity_search(query, k=k*2)):
            doc_id = f"{doc.metadata.get('title')}_{doc.page_content[:50]}"
            semantic_score = 1.0 / (i + 1)
            if doc_id in combined:
                combined[doc_id]['semantic'] = semantic_score
            else:
                combined[doc_id] = {'doc': doc, 'bm25': 0, 'semantic': semantic_score}
        
        bm25_w = float(os.getenv("BM25_WEIGHT", "0.5"))
        semantic_w = float(os.getenv("SEMANTIC_WEIGHT", "0.5"))
        
        for item in combined.values():
            item['score'] = item['bm25'] * bm25_w + item['semantic'] * semantic_w
        
        return [item['doc'] for item in sorted(combined.values(), key=lambda x: x['score'], reverse=True)[:k]]

    def build_chain(self):
        # Create the LangChain pipeline that connects prompt to LLM
        prompt = ChatPromptTemplate.from_template(
            "You are a helpful movie assistant. Answer the user's question based on the provided movie plot contexts. "
            "Respond in JSON format with these exact fields: answer, contexts (as a list), and reasoning.\n\n"
            "CONTEXTS:\n{context}\n\n"
            "QUESTION:\n{question}"
        )
        self.chain = prompt | self.llm
        print("RAG chain built.")

    def query(self, question: str) -> Dict[str, Any]:
        # Search for relevant movie plots and generate answer with reasoning
        print(f"\nProcessing: '{question}'")
        if not self.chain:
            self.build_chain()
        
        docs = self.hybrid_search(question)
        context = "\n\n---\n\n".join(
            f"Movie: {d.metadata.get('title')} ({d.metadata.get('year')})\n{d.page_content}"
            for d in docs
        )
        
        # Invoke chain and get result
        result = self.chain.invoke({"context": context, "question": question})
        
        return {
            "answer": result.answer,
            "contexts": result.contexts,
            "reasoning": result.reasoning
        }

def main():
    # Initialize the RAG system and start interactive Q&A loop
    rag = RAGSystem()
    rag.load_and_process_data()
    rag.build_chain()
    
    print("\n" + "="*50)
    print("RAG System Ready!")
    print("="*50)
    
    try:
        while True:
            question = input("\nQuestion (or 'quit'): ").strip()
            if question.lower() in ['quit', 'exit', 'q'] or not question:
                break
            
            try:
                print("\n" + "-"*50)
                print(json.dumps(rag.query(question), indent=2))
                print("-"*50)
            except Exception as e:
                print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()

# save this as test_ollama.py and run it
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Test LLM
llm = Ollama(model="llama3.2:3b", base_url="http://localhost:11434")
response = llm.complete("What is RAG in one sentence?")
print("LLM response:", response)

# Test Embeddings
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434"
)
embedding = embed_model.get_text_embedding("test sentence")
print(f"Embedding dimension: {len(embedding)}")
print("Setup complete!")
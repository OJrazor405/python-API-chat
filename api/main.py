from fastapi import FastAPI
from requestModels import PDFRequest, PromptRequest
from embedder import Embedder
from vectordb import VectorDB
from semantic_search import SemanticSearch

app = FastAPI()

# Create instances of classes
embedding = Embedder()

@app.post('/python/pdf')
def process_pdf(request: PDFRequest): 
    try:
        texts = embedding.embedPdf(request.url)
        embeddings = embedding.embedder()

        vectordb = VectorDB(embeddings, request.namespace, request.index_name)
        # vectordb.deleteAllVectors()
        vectordb.pushTextEmbeddings(texts)

        return {"message": "File successfully uploaded"}, 200
    except Exception as e:
        print(str(e))
        return { "error": str(e) }, 500

@app.post('/python/prompt')
def process_prompt(request: PromptRequest):    
    try:
        embeddings = embedding.embedder()
        vectordb = VectorDB(embeddings, request.namespace, request.index_name) # Assuming you need to connect to the VectorDB.
        vectorstore = vectordb.pullVectorstore()
        
        semantic_search = SemanticSearch(vectorstore, request.prompt)
        answer = semantic_search.runSemanticSearch()

        return { "answer": answer }, 200

    except Exception as e:
        print(str(e))
        return { "error": str(e) }, 500

@app.get("/python/answer/{id}")
def get_answer(id: int):
    # Fetch the answer from storage or database based on ID.
    # Return response with the answer, if found.
    answer = ""
    if answer is None:
        return { "error": "No answer found with given id" }, 404
    else:
        return answer, 200

# You can also include other endpoints or configurations here.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

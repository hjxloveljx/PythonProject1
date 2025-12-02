from app.ingestion.loder import load_docs,split_docs
from app.deps import get_embeddings,get_vs

def main():
    docs = split_docs(load_docs(r"/home/hjx/PycharmProjects/PythonProject1/data/docs"))
    vs = get_vs()
    vs.add_documents(docs)
    try:
        vs.persist()
    except Exception:
        pass
    print(f"Indexed{len(docs)}chunks into Chroma.")
if __name__ == "__main__":
    main()
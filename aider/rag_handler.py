import os
from datetime import datetime

import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from litellm import client
from numpy.f2py.auxfuncs import throw_error
import chromadb

from aider.utils import Spinner


#vectorstore = None

def handle_rag_request(cache_dir: str, user_message) -> str:
    persistent_client = chromadb.PersistentClient(
        path=cache_dir,
        settings=chromadb.config.Settings(
            persist_directory=cache_dir,
        )
    )
    vectorstore = Chroma(
        persist_directory=cache_dir,
        client=persistent_client,
        collection_name="embeddings",
        embedding_function=OpenAIEmbeddings()
    )

    # Retrieve and generate using the relevant snippets of the blog.
    if vectorstore is not None:
        retriever = vectorstore.as_retriever()

        answer = _format_docs(retriever.invoke(user_message))

        print("RAG results: \n" + answer)
        return answer
    return ""

def _format_docs(docs):
    return "\n\n".join(doc.metadata["path"] + "\n" + doc.page_content for doc in docs)

def rag_index_codebase(cache_dir: str, paths: set):
    spinner = Spinner("Indexing RAG vector db")

    valid_paths = []
    docs = []
    for path in paths:
        spinner.step()
        if os.path.isfile(path) and not ".aider" in path:
            try:
                with open(path, 'r', encoding="utf-8") as f:
                    docs.append(Document(
                        page_content=f.read(),
                        metadata={
                            "path": path,
                            "last_modified": datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
                        },
                        id=path
                    ))
            except UnicodeDecodeError as e:
                print(f"Skipping {path}, because of error {e}")
                pass
            valid_paths.append(path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    path_ids = {}
    for doc in splits:
        spinner.step()
        if doc.id not in path_ids:
            path_ids[doc.id] = 1
        else:
            path_ids[doc.id] += 1
        doc.id = f"{doc.id}-{str(path_ids[doc.id]).zfill(3)}"

    print(f"\nFound {len(valid_paths)}/{len(paths)} valid files, split into {len(splits)} chunks\n")

    persistent_client = chromadb.PersistentClient(
        path=cache_dir,
        settings=chromadb.config.Settings(
            persist_directory=cache_dir,
        )
    )
    collection = persistent_client.get_or_create_collection("embeddings")

    vectorstore = Chroma(
        persist_directory=cache_dir,
        client=persistent_client,
        collection_name="embeddings",
        embedding_function=OpenAIEmbeddings()
    )
    documents_to_add = []
    documents_to_update = [[], []]
    for split in splits:
        spinner.step()
        docs: list[Document] = vectorstore.get(split.id)["metadatas"]
        if len(docs) == 0:
            documents_to_add.append(split)
        else:
            if docs[0]["last_modified"] < split.metadata["last_modified"]:
                documents_to_update[0].append(split.id)
                documents_to_update[1].append(split)

    spinner.step()
    if len(documents_to_update[0]) > 0:
        print(f"\nUpdated {len(documents_to_update)}\n")
        vectorstore.update_documents(documents_to_update[0], documents_to_update[1])

    spinner.step()
    if len(documents_to_add) > 0:
        print(f"\nAdded {len(documents_to_add)}\n")
        batch_size = 5000
        batches = [documents_to_add[i:i+batch_size] for i in range(0, len(documents_to_add), batch_size)]
        for batch in batches:
            vectorstore.add_documents(batch)
    spinner.end()

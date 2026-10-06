from pathlib import Path
from typing import Any

import chromadb
import torch
from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM


class RAGEngine:

    def __init__(self, persist_dir: str):

        # -----------------------------
        # ChromaDB
        # -----------------------------
        self.client = chromadb.PersistentClient(
            path=str(Path(persist_dir))
        )

        self.collection = self.client.get_or_create_collection(
            "college_documents",
            metadata={"hnsw:space": "cosine"}
        )

        # -----------------------------
        # Embedding Model
        # -----------------------------
        self.embed_tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.embed_model = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.embed_model.eval()

        # -----------------------------
        # Answer Generation Model
        # -----------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            "google/flan-t5-small"
        )

        self.generator = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-small"
        )

        self.generator.eval()

    # =========================================================
    # CREATE EMBEDDINGS
    # =========================================================

    def create_embeddings(self, texts: list[str]) -> list[list[float]]:

        inputs = self.embed_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        with torch.inference_mode():

            outputs = self.embed_model(**inputs)

            token_embeddings = outputs.last_hidden_state

            attention_mask = inputs["attention_mask"]

            mask = attention_mask.unsqueeze(-1).expand(
                token_embeddings.size()
            ).float()

            summed = torch.sum(
                token_embeddings * mask,
                dim=1
            )

            counts = torch.clamp(
                mask.sum(dim=1),
                min=1e-9
            )

            embeddings = summed / counts

            embeddings = torch.nn.functional.normalize(
                embeddings,
                p=2,
                dim=1
            )

        return embeddings.cpu().tolist()

    # =========================================================
    # EXTRACT PDF TEXT
    # =========================================================

    @staticmethod
    def extract_pdf(path: Path) -> str:

        reader = PdfReader(str(path))

        text = []

        for page in reader.pages:

            page_text = page.extract_text() or ""

            text.append(page_text)

        return "\n".join(text).strip()

    # =========================================================
    # SPLIT TEXT INTO CHUNKS
    # =========================================================

    @staticmethod
    def chunks(
        text: str,
        size: int = 900,
        overlap: int = 150
    ) -> list[str]:

        text = " ".join(text.split())

        output = []

        start = 0

        while start < len(text):

            end = min(
                start + size,
                len(text)
            )

            chunk = text[start:end]

            # Try to end at a sentence
            if end < len(text):

                stop = max(
                    chunk.rfind(". "),
                    chunk.rfind("? "),
                    chunk.rfind("! ")
                )

                if stop > size * 0.55:

                    end = start + stop + 1

                    chunk = text[start:end]

            if len(chunk.strip()) >= 40:

                output.append(
                    chunk.strip()
                )

            if end >= len(text):
                break

            start = max(
                end - overlap,
                start + 1
            )

        return output

    # =========================================================
    # ADD PDF TO KNOWLEDGE BASE
    # =========================================================

    def add_pdf(self, path: Path) -> bool:

        text = self.extract_pdf(path)

        chunks = self.chunks(text)

        if not chunks:
            return False

        name = path.name

        ids = [
            f"{name}::chunk::{i}"
            for i in range(len(chunks))
        ]

        # Remove old version of the same PDF
        try:
            self.collection.delete(ids=ids)
        except Exception:
            pass

        # Create embeddings
        embeddings = self.create_embeddings(chunks)

        metadata = [
            {
                "source": name,
                "chunk": i
            }
            for i in range(len(chunks))
        ]

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadata
        )

        return True

    # =========================================================
    # RETRIEVE RELEVANT DOCUMENTS
    # =========================================================

    def retrieve(
        self,
        query: str,
        k: int = 4
    ) -> list[dict[str, Any]]:

        count = self.collection.count()

        if not count:
            return []

        query_embedding = self.create_embeddings(
            [query]
        )

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(k, count)
        )

        documents = results.get("documents", [[]])[0]

        metadatas = results.get("metadatas", [[]])[0]

        return [
            {
                "text": document,
                "source": metadata.get(
                    "source",
                    "Unknown"
                ),
                "chunk": metadata.get(
                    "chunk",
                    "?"
                )
            }

            for document, metadata
            in zip(documents, metadatas)
        ]

    # =========================================================
    # GENERATE ANSWER
    # =========================================================

    def answer(self, query: str) -> dict[str, Any]:

        documents = self.retrieve(query)

        if not documents:

            return {
                "answer": (
                    "No college documents are available. "
                    "Upload PDFs and click Add PDFs."
                ),
                "sources": []
            }

        # Build context
        context = "\n\n".join(
            f"[Source: {doc['source']} | "
            f"Chunk: {doc['chunk']}]\n"
            f"{doc['text']}"
            for doc in documents
        )

        prompt = f"""
You are an AI college assistant.

Answer the question using ONLY the information
provided in the context below.

Do not invent facts.

If the answer is not available in the context,
say:

The information was not found in the uploaded college documents.

Keep the answer simple and easy to understand.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        # Generate answer
        with torch.inference_mode():

            outputs = self.generator.generate(
                **inputs,
                max_new_tokens=220,
                do_sample=False
            )

        answer_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        ).strip()

        return {
            "answer": answer_text,
            "sources": [
                f"{doc['source']} — chunk {doc['chunk']}"
                for doc in documents
            ]
        }

    # =========================================================
    # CLEAR DATABASE
    # =========================================================

    def clear_database(self):

        try:

            self.client.delete_collection(
                "college_documents"
            )

        except Exception:

            pass

        self.collection = self.client.get_or_create_collection(
            "college_documents",
            metadata={"hnsw:space": "cosine"}
        )
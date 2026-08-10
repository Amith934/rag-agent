import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.1
)

GUARDRAIL_PROMPT = """You are a knowledge-base assistant.
Answer ONLY using the retrieved context below from the uploaded document.
If the answer cannot be found in the retrieved context, respond exactly:
"I don't know based on the uploaded knowledge base."
Do not use outside knowledge. Do not guess. Do not invent facts.

Retrieved context:
{context}

Question: {question}

Answer:"""


def extract_text(response) -> str:
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content)


def answer_question(vectorstore, question: str, top_k: int = 4) -> dict:
    results = vectorstore.similarity_search_with_score(question, k=top_k)

    if not results:
        return {
            "answer": "I don't know based on the uploaded knowledge base.",
            "sources": []
        }

    context = "\n\n---\n\n".join(doc.page_content for doc, score in results)
    prompt = GUARDRAIL_PROMPT.format(context=context, question=question)
    response = llm.invoke(prompt)
    answer = extract_text(response)

    sources = [
        {"page": str(doc.metadata.get("page", "?")), "snippet": doc.page_content[:150]}
        for doc, score in results
    ]

    return {"answer": answer, "sources": sources}
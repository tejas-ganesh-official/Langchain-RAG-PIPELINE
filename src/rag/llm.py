"""LLM: local Qwen model served through Ollama."""

from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


class QwenLLM:
    """Wrapper around a local Qwen model served via Ollama."""

    def __init__(
        self,
        model_name: str = "qwen2.5:3b",
        base_url: str = "http://localhost:11434",
    ):
        """Initialize local Qwen LLM via Ollama.

        Args:
            model_name: Ollama model tag (must match what you pulled,
                e.g. 'qwen2.5:3b').
            base_url: Ollama server URL (default local instance).
        """
        self.model_name = model_name
        self.base_url = base_url

        self.llm = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=0.1,
            num_predict=1024,  # Ollama's equivalent of max_tokens
        )

        print(f"Initialized Qwen LLM (Ollama) with model: {self.model_name}")

    def generate_response(self, query: str, context: str, max_length: int = 500) -> str:
        """Answer a question grounded in the provided context."""
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a helpful AI assistant. Use the following context to "
                "answer the question accurately and concisely.\n"
                "Context:\n{context}\n"
                "Question: {question}\n"
                "Answer: Provide a clear and informative answer based on the "
                "context above. If the context doesn't contain enough "
                "information to answer the question, say so."
            ),
        )

        formatted_prompt = prompt_template.format(context=context, question=query)

        try:
            messages = [HumanMessage(content=formatted_prompt)]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def generate_response_simple(self, query: str, context: str) -> str:
        """A minimal prompt variant for quick answers."""
        simple_prompt = f"""Based on this context: {context}
Question: {query}
Answer:"""

        try:
            messages = [HumanMessage(content=simple_prompt)]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"


def build_qwen_llm(model_name: str = "qwen2.5:3b"):
    """Factory that initializes QwenLLM and fails gracefully.

    Returns the QwenLLM instance, or ``None`` if Ollama isn't reachable.
    """
    try:
        qwen_llm = QwenLLM(model_name=model_name)
        print("Qwen LLM initialized successfully!")
        return qwen_llm
    except Exception as e:
        print(f"Warning: {e}")
        print(
            "Make sure Ollama is running (`ollama serve`) and the model is "
            "pulled (`ollama pull qwen2.5:3b`)."
        )
        return None

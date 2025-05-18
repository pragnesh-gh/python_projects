from typing import Dict, Any
import json
import openai
import ollama

class BaseAgent: #all agents are going to ingherit from this base agent, they are going to be modeled after this.
    # We are using Swarm framework here and overriding the OpenAI stuff to use our local stuff.
    def __init__(self, name: str, instructions: str):
        self.name = name
        self.instructions = instructions
        # self.ollama_client = OpenAI(
        #     base_url="http://localhost:11434/v1", #here we are mentioning the local URL, overwriting the open Ai one.
        #     api_key="ollama",  # required but unused by Ollama
        # )
        ollama.pull("gemma3:4b")
      # # Point the OpenAI client at your local Ollama server:
      #   openai.api_base = "http://localhost:11434/v1"
      #   openai.api_key = None
      #   self.ollama_client = openai  # we'll call ChatCompletion on this

    async def run(self, messages: list) -> Dict[str, Any]:
        """Default run method to be overridden by child classes"""
        raise NotImplementedError("Subclasses must implement run()")

    def _query_ollama(self, prompt: str) -> str:
        """Query the Ollama model with system+user, plus temperature and max tokens."""
        try:
            # Ollama’s Python API exposes both `chat` and `generate`.
            # Here we use `generate` (it returns {"response": ...})
            result = ollama.generate(
                model="gemma3:4b",
                prompt=(
                    f"SYSTEM: {self.instructions}\n\n"
                    f"USER: {prompt}"
                ),
                options={
                    "temperature": 0.7,    # same as before
                    "num_predict": 2000,   # your "max_tokens"
                },
            )
            return result["response"]

        except Exception as e:
            # If anything goes wrong, print & re‐raise so your orchestrator can catch it
            print(f"Error querying Ollama: {e}")
            raise

    def _parse_json_safely(self, text: str) -> Dict[str, Any]:
        """Safely parse JSON from text, handling potential errors"""
        try:
            # Try to find JSON-like content between curly braces
            start = text.find("{")
            end   = text.rfind("}")
            if start != -1 and end != -1:
                json_str = text[start : end + 1]
                return json.loads(json_str)
            return {"error": "No JSON content found"}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON content"}

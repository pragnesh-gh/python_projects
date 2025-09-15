from typing import Dict, Any
from pdfminer.high_level import extract_text    # pip install pdfminer.six
from .base_agent import BaseAgent

class ExtractorAgent(BaseAgent):
    def __init__(self):
        """
        Initialize the agent with a name and system‐level instructions
        that tell Ollama _how_ to extract and structure information.
        """
        super().__init__(
            name="Extractor",
            instructions=(
                "Extract and structure information from resumes.\n"
                "Focus on: personal info, work experience, education, skills, and certifications.\n"
                "Provide output in a clear, structured format."
            )
        )

    async def run(self, messages: list) -> Dict[str, Any]:
        """
        1. Pull the raw payload from the last message in the convo.
        2. Decide whether to read PDF from disk or use provided text.
        3. Call Ollama (_query_ollama) to convert raw text → structured JSON-like data.
        4. Return both raw and structured outputs along with a status.
        """
        print("📄 Extractor: Processing resume")

        # The last message’s content is expected to be a Python‐literal dict string,
        # e.g. "{'file_path': 'data/resume.pdf'}" or "{'text': 'Alice worked at…'}"
        resume_data = eval(messages[-1]["content"])

        # 1) If `file_path` is provided, extract text from the PDF on disk:
        if resume_data.get("file_path"):
            raw_text = extract_text(resume_data["file_path"])
        else:
            # 2) Otherwise, fall back to a direct `text` field (could be a plain‐text resume)
            raw_text = resume_data.get("text", "")

        # 3) Send the raw text into your base class’s Ollama wrapper
        #    which will apply your `instructions` and return the parsed result
        extracted_info = self._query_ollama(raw_text)

        # 4) Return both the unmodified "raw_text" and the LLM‐parsed structure
        return {
            "raw_text": raw_text,
            "structured_data": extracted_info,
            "extraction_status": "completed",
        }

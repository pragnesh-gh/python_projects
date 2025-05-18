from typing import Dict, Any
from .base_agent import BaseAgent

#inherinting from the base agent, which has all the properties and methods for it to tbecome an agent.
class ScreenerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Screener",
            instructions="""Screen candidates based on:
            - Qualification alignment
            - Experience relevance
            - Skill match percentage
            - Cultural fit indicators
            - Red flags or concerns
            Provide comprehensive screening reports."""
        )

    async def run(self, messages: list) -> Dict[str, Any]:
        """Screen the candidate"""
        print("🕵️‍♂️ Screener: Conducting initial screening")

        # The last message in `messages` contains the workflow context as a JSON-like string
        workflow_context = eval(messages[-1]["content"])

        # Convert that context to string and send to Ollama for screening analysis
        screening_results = self._query_ollama(str(workflow_context))

        return {
            "screening_report": screening_results,
            "screening_timestamp": "2024-03-14",
            "screening_score": 85,
        }

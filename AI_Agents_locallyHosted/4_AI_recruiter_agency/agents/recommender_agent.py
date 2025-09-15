from typing import Dict, Any
from .base_agent import BaseAgent

class RecommenderAgent(BaseAgent):
    def __init__(self):
        """
        1) Give this agent a name (“Recommender”)
        2) Provide system‐level instructions that tell Ollama how to
           synthesize all the upstream data into final advice.
        """
        super().__init__(
            name="Recommender",
            instructions=(
                "Generate final recommendations considering:\n"
                "1. Extracted profile\n"
                "2. Skills analysis\n"
                "3. Job matches\n"
                "4. Screening results\n"
                "Provide clear next steps and specific recommendations."
            ),
        )

    async def run(self, messages: list) -> Dict[str, Any]:
        """
        1. Pull the entire workflow context (a dict‐string) from the last message
        2. Convert it back into a Python dict via eval()
        3. Stringify that dict and feed it to Ollama via _query_ollama()
        4. Wrap Ollama’s reply up with a timestamp & confidence level
        5. Return the final dictionary
        """
        print("💡 Recommender: Generating final recommendations")

        # The previous agent packed everything into messages[-1]["content"]
        workflow_context = eval(messages[-1]["content"])

        # Ask Ollama to combine profile, analysis, matches, screening → next steps
        recommendation = self._query_ollama(str(workflow_context))

        # Standardize our output schema for downstream use
        return {
            "final_recommendation": recommendation,
            "recommendation_timestamp": "2024-03-14",
            "confidence_level": "high",
        }

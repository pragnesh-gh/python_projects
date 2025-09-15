from typing import Dict, Any

# Import each specialized agent
from .base_agent      import BaseAgent
from .extractor_agent import ExtractorAgent
from .analyzer_agent  import AnalyzerAgent
from .matcher_agent   import MatcherAgent
from .screener_agent  import ScreenerAgent
from .recommender_agent import RecommenderAgent

class OrchestratorAgent(BaseAgent):
    def __init__(self):
        """
        1) Name this agent “Orchestrator”
        2) Give it high-level instructions about coordination
        3) Instantiate all the stage-specific agents
        """
        super().__init__(
            name="Orchestrator",
            instructions=(
                "Coordinate the recruitment workflow and delegate tasks "
                "to specialized agents. Ensure proper flow of information "
                "between extraction, analysis, matching, screening, and "
                "recommendation stages. Maintain context and aggregate "
                "results from each stage."
            ),
        )
        self._setup_agents()

    def _setup_agents(self):
        """Create one instance of each specialized agent."""
        self.extractor   = ExtractorAgent()
        self.analyzer    = AnalyzerAgent()
        self.matcher     = MatcherAgent()
        self.screener    = ScreenerAgent()
        self.recommender = RecommenderAgent()

    async def run(self, messages: list) -> Dict[str, Any]:
        """
        A simple passthrough for a single prompt:
          • Takes last user message
          • Calls Ollama directly with that prompt
          • Parses JSON safely
        """
        prompt   = messages[-1]["content"]
        response = self._query_ollama(prompt)
        return self._parse_json_safely(response)

    async def process_application(self, resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        End-to-end workflow:
          1. Start with raw resume data
          2. Extract → Analyze → Match → Screen → Recommend
          3. Update `workflow_context` at each stage
          4. Return the full context when done
        """
        print("🛠️  Orchestrator: Starting application process")

        # 1) Initialize our shared context
        workflow_context = {
            "resume_data": resume_data,
            "status":      "initiated",
            "current_stage": "extraction",
        }

        try:
            # 2) Extraction
            extracted_data = await self.extractor.run([
                {"role": "user", "content": str(resume_data)}
            ])
            workflow_context.update({
                "extracted_data": extracted_data,
                "current_stage": "analysis"
            })

            # 3) Analysis
            analysis_results = await self.analyzer.run([
                {"role": "user", "content": str(extracted_data)}
            ])
            workflow_context.update({
                "analysis_results": analysis_results,
                "current_stage":    "matching"
            })

            # 4) Matching
            job_matches = await self.matcher.run([
                {"role": "user", "content": str(analysis_results)}
            ])
            workflow_context.update({
                "job_matches": job_matches,
                "current_stage": "screening"
            })

            # 5) Screening
            screening_results = await self.screener.run([
                {"role": "user", "content": str(workflow_context)}
            ])
            workflow_context.update({
                "screening_results": screening_results,
                "current_stage":     "recommendation"
            })

            # 6) Final Recommendation
            final_recommendation = await self.recommender.run([
                {"role": "user", "content": str(workflow_context)}
            ])
            workflow_context.update({
                "final_recommendation": final_recommendation,
                "status":               "completed"
            })

            return workflow_context

        except Exception as e:
            # On any error, record failure and re-raise
            workflow_context.update({
                "status": "failed",
                "error":  str(e)
            })
            raise

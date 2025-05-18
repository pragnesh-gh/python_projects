from typing import Dict, Any
from .base_agent import BaseAgent

class MatcherAgent(BaseAgent):
    def __init__(self):
        """
        Initialize the matcher with:
          • A name (“Matcher”)
          • System-level instructions that tell Ollama how to perform the matching.
        """
        super().__init__(
            name="Matcher",
            instructions=(
                "Match candidate profiles with job positions.\n"
                "Consider: skills match, experience level, location preferences.\n"
                "Provide detailed reasoning and compatibility scores.\n"
                "Return matches in JSON format with title, match_score, and location fields."
            )
        )

    async def run(self, messages: list) -> Dict[str, Any]:
        """
        1. Pull the analyzed profile (a dict-string) from the last message.
        2. Define a list of sample jobs to match against.
        3. Prompt Ollama to rank+score those jobs for this profile.
        4. Parse the JSON response safely.
        5. If parsing fails, fall back to our static sample data.
        6. Return the final matches structure.
        """
        print("🎯 Matcher: Finding suitable job matches")

        # 1) Convert last message’s content into a Python dict:
        analysis_results = eval(messages[-1]["content"])
        #    Expect keys like "skills_analysis", etc., from previous agent.

        # 2) Define some example jobs (in practice, you'd fetch these dynamically):
        #Maybe an API that has many different jobs or a PDF or csv file.
        #Pluggin in the source where we will have different jobs, requirements, locations...
        sample_jobs = [
            {
                "title": "Senior Software Engineer",
                "requirements": "Python, Cloud, 5+ years experience",
                "location": "Remote",
            },
            {
                "title": "Data Scientist",
                "requirements": "Python, ML, Statistics, 3+ years experience",
                "location": "New York",
            },
        ]

        # 3) Ask Ollama to analyze & score the matches
        matching_response = self._query_ollama(
            f"""Analyze the following profile and provide job matches in valid JSON format.
            Profile skills: {analysis_results.get('skills_analysis')}
            Available Jobs: {sample_jobs}
            
            Return ONLY a JSON object with this exact structure:
            {{
              "matched_jobs": [
                {{
                  "title": "job title",
                  "match_score": "85%",
                  "location": "job location"
                }}
              ],
              "match_timestamp": "2024-03-14",
              "number_of_matches": 2
            }}
            """
        )

        # 4) Parse the returned JSON safely using the function from the base agent
        parsed_response = self._parse_json_safely(matching_response)

        # 5) Fallback if parsing fails
        if "error" in parsed_response:
            return {
                "matched_jobs": [
                    {"title": "Senior Software Engineer", "match_score": "85%", "location": "Remote"},
                    {"title": "Data Scientist", "match_score": "75%", "location": "New York"},
                ],
                "match_timestamp": "2024-03-14",
                "number_of_matches": 2,
            }

        # 6) Normal case: return what the LLM gave us
        return parsed_response

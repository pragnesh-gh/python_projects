from typing import Dict, Any
from swarm import Agent

def profile_enhancer_agent_function(extracted_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    1) Takes the raw `extracted_info` dict from the ExtractorAgent
    2) Computes total years of experience
    3) Adds a human-readable summary field
    4) Returns the “enhanced” profile dict
    """
    # Start by copying the incoming dict so we don’t mutate the original
    enhanced_profile = extracted_info.copy()

    # Sum up all the “years” values in the experience list (if any)
    total_experience_years = sum(
        item.get("years", 0)
        for item in extracted_info.get("experience", [])
    )

    # Inject a new summary field into the profile
    enhanced_profile["summary"] = (
        f"{extracted_info.get('name', 'The candidate')} "
        f"has {total_experience_years} years of experience "
        "in relevant roles."
    )

    return enhanced_profile


# Now wrap that function into a Swarm Agent
profile_enhancer_agent = Agent(
    name="Profile Enhancer Agent",
    model="gemma3:4b",
    instructions=(
        "Enhance the candidate’s profile based on the extracted information. "
        "Add any high-level insights or summaries that will help downstream steps."
    ),
    functions=[profile_enhancer_agent_function],
)

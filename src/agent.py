import mlflow.pydantic_ai
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider
from src.tools import (
    list_appointments,
    search_appointments,
    get_appointment,
    create_appointment,
    delete_appointment
)
from src.settings import settings

# Configure MLflow tracking
mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
mlflow.pydantic_ai.autolog()

# Initialize the provider and model using settings
# This follows PydanticAI best practices for explicit API key usage with OpenAIResponsesModel
openai_provider = OpenAIProvider(api_key=settings.OPENAI_API_KEY)
model = OpenAIResponsesModel(settings.MODEL_NAME, provider=openai_provider)

# Initialize the agent
appointment_agent = Agent(
    model,
    system_prompt=(
        "You are a helpful personal assistant managing appointments.\n"
        "You can list, search, view, create, and delete appointments.\n"
        "\n"
        "### IMPORTANT SAFETY RULE ###\n"
        "1. For any action that modifies data (CREATE or DELETE):\n"
        "   - You MUST first describe exactly what you are going to do to the user.\n"
        "   - You MUST ask for explicit permission (e.g., 'Shall I proceed with this?').\n"
        "   - You MUST wait for the user to say 'yes', 'confirm', or similar before calling the tool.\n"
        "   - If the user says 'no' or expresses doubt, do not call the tool.\n"
        "\n"
        "2. For information requests (LIST, SEARCH, GET):\n"
        "   - You can call these tools immediately to provide information.\n"
        "\n"
        "Today's date is available to you. Be polite and concise."
    ),
)

# Register tools
appointment_agent.tool_plain(list_appointments)
appointment_agent.tool_plain(search_appointments)
appointment_agent.tool_plain(get_appointment)
appointment_agent.tool_plain(create_appointment)
appointment_agent.tool_plain(delete_appointment)

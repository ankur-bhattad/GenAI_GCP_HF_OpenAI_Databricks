# Step 1: Set Up API Key and Environment
import os
from dotenv import load_dotenv
import autogen
from openai import AzureOpenAI, APIError, Timeout  # Use AzureOpenAI directly

# Load environment variables from .env
load_dotenv("E:\\Lesson_3_demos\\.env")

AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT_NAME")  # your deployment

# print("API KEY:", os.getenv("AZURE_OPENAI_API_KEY"))
# print("ENDPOINT:", os.getenv("AZURE_OPENAI_ENDPOINT"))
# print("DEPLOYMENT:", os.getenv("AZURE_DEPLOYMENT_NAME"))

if not AZURE_API_KEY or not AZURE_ENDPOINT or not AZURE_DEPLOYMENT:
    raise ValueError(
        "Set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_DEPLOYMENT_NAME in .env"
    )

# Step 2: Create Customer Agent
customer_agent = autogen.UserProxyAgent(
    name="customer",
    human_input_mode="ALWAYS",  # Allows manual input
    code_execution_config={"use_docker": False},
    max_consecutive_auto_reply=5
)

# Step 3: Create Support Agent using Azure OpenAI
support_agent = autogen.AssistantAgent(
    name="support_agent",
    llm_config={
        "config_list": [
            {
                "api_type": "azure",
                "api_key": AZURE_API_KEY,
                "api_version": AZURE_API_VERSION,
                "azure_endpoint": AZURE_ENDPOINT,
                "model": AZURE_DEPLOYMENT
            }
        ],
        "temperature": 0.7,
    },
    system_message="You are a helpful AI support agent. Answer customer queries clearly and professionally.",
    code_execution_config={"use_docker": False},
    max_consecutive_auto_reply=5
)

# Step 4: Run a simulated customer interaction safely
try:
    customer_agent.initiate_chat(support_agent, message="I need help tracking my order.")
except APIError as e:
    print("API error:", e)
except Timeout as e:
    print("Request timed out:", e)
except Exception as e:
    print("Unexpected error:", e)

import json
import os

from langchain_community.chat_models import ChatHuggingFace
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.tools.json.tool import JsonSpec
from langchain.agents import create_json_agent
from langchain.agents.agent_toolkits.json.toolkit import JsonToolkit

# ------------------------------------------------------------------------------
# Initialize Variables - DO NOT TOUCH
# ------------------------------------------------------------------------------

json_path = "/resources/example.json"
questions = [
    "What does this json file contain?",
    "What endpoints are available to us?",
    "Which endpoint is the most useful?",
]

llm = HuggingFaceEndpoint(
        endpoint_url="https://z8dvl7fzhxxcybd8.eu-west-1.aws.endpoints.huggingface.cloud",
        task="text2text-generation",
        model_kwargs={
            "max_new_tokens": 200
        }
    )
chat_model = ChatHuggingFace(llm=llm)

with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '../resources/example.json')),
          encoding="utf8") as file:
    data = json.load(file)

# ------------------------------------------------------------------------------
# TODO Functions - Implement the logic as per instructions
# ------------------------------------------------------------------------------


def execute_json_agent():
    """
    TODO: This function executes a JSON agent using the langchain JSON toolkit, using the list of questions in the list
        above. It then returns all responses from the JSON agent as a list.

    Instructions:
    - Initialize an empty list named 'response' to store the responses from the JSON agent.
    - Create a JsonToolkit instance named 'json_toolkit'. The JsonSpec class is used to create a specification for the
        toolkit using the 'data' dictionary and a maximum value length of 4000.
    - Create a JSON agent using the 'create_json_agent' function. The 'chat' instance of AzureChatOpenAI, the
        'json_toolkit', and a verbosity setting of True are passed as arguments.
    - Loop through the 'questions' list. For each question, the 'run' method of the agent is called with the question as an argument. The response is appended to the 'response' list.
    - Return the 'response' list.

    :return: A list of responses from the JSON agent.
    """
    # Write Code Below
    response = []
    json_toolkit = JsonToolkit(spec=JsonSpec(dict_=data, max_value_length=1000))
    agent = create_json_agent(llm=chat_model, toolkit=json_toolkit, verbose=True)
    for question in questions:
        response.append(agent.run(question))
    return response

    # Replace with return statement
    # raise NotImplementedError("This function has not been implemented yet.")

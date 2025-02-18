from autogen.agentchat import UserProxyAgent , AssistantAgent, GroupChat, GroupChatManager
import os
from dotenv import load_dotenv
load_dotenv()

config_list_gpt4 = [
    {
        "model": "gpt-4o",
        "api_type": "azure",
        "api_key": os.getenv('AZURE_OPENAI_KEY'),
        "base_url": os.getenv('AZURE_OAI_ENDPOINT'),
        "api_version": "2024-06-01"
    },
]
#if you are uisng openai api key, use the below config:
#config_list_gpt4 = [{"model": "gpt-4o", "api_key": os.getenv('OPENAI_API_KEY')}]

gpt4_config = {
    "cache_seed":42,
    "temperature":0,
    "config_list":config_list_gpt4,
    "timeout":120,
}

# User Proxy Agent  
user_proxy = UserProxyAgent(  
    name="Admin",  
    human_input_mode="ALWAYS",  
    system_message="1. A human admin. 2. Interact with the team. 3. Plan execution needs to be approved by this Admin.",  
    code_execution_config=False,  
    llm_config=gpt4_config,  
    description="""Call this Agent if:   
        You need guidance.
        The program is not working as expected.
        You need api key                  
        DO NOT CALL THIS AGENT IF:  
        You need to execute the code.""",  
)  
  
# Assistant Agent - Developer  
developer = AssistantAgent(  
    name="Developer",  
    llm_config=gpt4_config,  
    system_message="""You are an AI developer. You follow an approved plan, follow these guidelines: 
    1. You write python/shell code to solve tasks. 
    2. Wrap the code in a code block that specifies the script type.   
    3. The user can't modify your code. So do not suggest incomplete code which requires others to modify.   
    4. You should print the specific code you would like the executor to run.
    5. Don't include multiple code blocks in one response.   
    6. If you need to import libraries, use ```bash pip install module_name```, please send a code block that installs these libraries and then send the script with the full implementation code 
    7. Check the execution result returned by the executor,  If the result indicates there is an error, fix the error and output the code again  
    8. Do not show appreciation in your responses, say only what is necessary.    
    9. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
    """,  
    description="""Call this Agent if:   
        You need to write code.                  
        DO NOT CALL THIS AGENT IF:  
        You need to execute the code.""",  
)  
# Assistant Agent - Planner  
planner = AssistantAgent(  
    name="Planner",  #2. The research should be executed with code
    system_message="""You are an AI Planner,  follow these guidelines: 
    1. Your plan should include 5 steps, you should provide a detailed plan to solve the task.
    2. Post project review isn't needed. 
    3. Revise the plan based on feedback from admin and quality_assurance.   
    4. The plan should include the various team members,  explain which step is performed by whom, for instance: the Developer should write code, the Executor should execute code, important do not include the admin in the tasks e.g ask the admin to research.  
    5. Do not show appreciation in your responses, say only what is necessary.  
    6. The final message should include an accurate answer to the user request
    """,  
    llm_config=gpt4_config,  
    description="""Call this Agent if:   
        You need to build a plan.                  
        DO NOT CALL THIS AGENT IF:  
        You need to execute the code.""",  
)  
  
# User Proxy Agent - Executor  
executor = UserProxyAgent(  
    name="Executor",  
    system_message="1. You are the code executer. 2. Execute the code written by the developer and report the result.3. you should read the developer request and execute the required code",  
    human_input_mode="NEVER",  
    code_execution_config={  
        "last_n_messages": 20,  
        "work_dir": "dream",  
        "use_docker": True,  
    },  
    description="""Call this Agent if:   
        You need to execute the code written by the developer.  
        You need to execute the last script.  
        You have an import issue.  
        DO NOT CALL THIS AGENT IF:  
        You need to modify code""",
)
quality_assurance = AssistantAgent(
    name="Quality_assurance",
    system_message="""You are an AI Quality Assurance. Follow these instructions:
      1. Double check the plan, 
      2. if there's a bug or error suggest a resolution
      3. If the task is not solved, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach.""",
    llm_config=gpt4_config,
)

allowed_transitions = {
    user_proxy: [ planner,quality_assurance],
    planner: [ user_proxy, developer, quality_assurance],
    developer: [executor,quality_assurance, user_proxy],
    executor: [developer],
    quality_assurance: [planner,developer,executor,user_proxy],
}

system_message_manager="You are the manager of a research group your role is to manage the team and make sure the project is completed successfully."
groupchat = GroupChat(
    agents=[user_proxy, developer, planner, executor, quality_assurance],allowed_or_disallowed_speaker_transitions=allowed_transitions,
    speaker_transitions_type="allowed", messages=[], max_round=30,send_introductions=True
)
manager = GroupChatManager(groupchat=groupchat, llm_config=gpt4_config, system_message=system_message_manager)

#here we print a graph representation of the code

task1="what are the 5 leading GitHub repositories on llm for the legal domain?"
chat_result=user_proxy.initiate_chat(
    manager,
    message=task1, 
    clear_history=True
)

task2="based on techcrunch, please find 3 articles on companies developing llm for legal domain, that rasied seed round. please use serper api"
chat_result=user_proxy.initiate_chat(
    manager,
    message=task2, 
    clear_history=False
)

import pprint
pprint.pprint(chat_result.cost)
#pprint.pprint(chat_result.summary)
#pprint.pprint(chat_result.chat_history)

for agent in groupchat.agents:
    agent.reset()
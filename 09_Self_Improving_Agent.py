# Key Components
# Language Model: The core of the agent, responsible for generating responses and processing information.
# Chat History Management: Keeps track of conversations for context and learning.
# Response Generation: Produces relevant replies to user inputs.
# Reflection Mechanism: Analyzes past interactions to identify areas for improvement.
# Learning System: Incorporates insights from reflection to enhance future performance.

from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import os
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

def get_chat_history(store, session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def generate_response(chain_with_history, human_input:str, session_id:str, insights:str):
    response = chain_with_history.invoke(
        {"input":human_input, "insights":insights},
        config = {"configurable":{"session_id":session_id}}
    )
    return response.content

def reflect(llm, store, session_id:str):
    reflection_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Based on the following conversation history, provide insights on how to improve responses:"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "Generate insights for improvement:")
        ]
    )
    reflection_chain = reflection_prompt | llm
    history = get_chat_history(store, session_id)
    reflection_response = reflection_chain.invoke({"history":history.messages})
    return reflection_response.content

def learn(llm, store, session_id:str, insights:str):
    learning_prompt = ChatPromptTemplate.from_messages([
        ("system", "Based on these insights, update the agent's knowledge and behavior:"),
        ("human", "{insights}"),
        ("human", "Summarize the key points to remember:")
    ])
    learning_chain = learning_prompt | llm
    learned_points = learning_chain.invoke({"insights": insights}).content
    get_chat_history(store, session_id).add_ai_message(f"[SYSTEM] Agent learned: {learned_points}")
    return learned_points

# Self-Improving Agent Class

class SelfImproveAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini",max_tokens=1000, temperature=0.7)
        self.store = {}
        self.insights = ""

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a self-improving AI assistant. Learn from your interactions and improve your performance over time."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
            ("system", "Recent insights for improvement: {insights}")
        ])

        self.chain = self.prompt | self.llm
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            lambda session_id: get_chat_history(self.store, session_id),
            input_messages_key="input",
            history_messages_key="history"
        )

    def respond(self, human_input: str, session_id: str):
        return generate_response(self.chain_with_history, human_input, session_id, self.insights)

    def reflect(self, session_id: str):
        self.insights = reflect(self.llm, self.store, session_id)
        return self.insights

    def learn(self, session_id: str):
        self.reflect(session_id)
        return learn(self.llm, self.store, session_id, self.insights)
    
agent = SelfImproveAgent()
session_id = "user_123"

# Interaction 1
print("AI:", agent.respond("What's the capital of France?", session_id))

# Interaction 2
print("AI:", agent.respond("Can you tell me more about its history?", session_id))

# Learn and improve
print("\nReflecting and learning...")
learned = agent.learn(session_id)
print("Learned:", learned)

# Interaction 3 (potentially improved based on learning)
print("\nAI:", agent.respond("What's a famous landmark in this city?", session_id))

# Interaction 4 (to demonstrate continued improvement)
print("AI:", agent.respond("What's another interesting fact about this city?", session_id))

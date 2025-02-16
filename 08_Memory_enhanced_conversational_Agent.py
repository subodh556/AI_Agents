# The Memory-Enhanced Conversational Agent offers several advantages over traditional chatbots:

# Improved Context Awareness: By utilizing both short-term and long-term memory, the agent can maintain context within and across conversations.
# Personalization: Long-term memory allows the agent to remember user preferences and past interactions, enabling more personalized responses.
# Flexible Memory Management: The implementation allows for easy adjustment of what information is stored long-term and how it's used in conversations.
# Scalability: The session-based approach allows for managing multiple independent conversations.

from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
# Initialize the language model
llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1000, temperature=0)

chat_store = {}
long_term_memory = {}

def get_chat_history(session_id:str):
    if session_id not in chat_store:
        chat_store[session_id] = ChatMessageHistory()
    return chat_store[session_id]

def update_long_term_memory(session_id:str, input:str, output:str):
    if session_id not in long_term_memory:
        long_term_memory[session_id] = []
    if len(input)>20: # Simple logic: store inputs longer than 20 characters
        long_term_memory[session_id].append(f"User said: {input}")
    if len(long_term_memory[session_id]) > 5:  # Keep only last 5 memories
        long_term_memory[session_id] = long_term_memory[session_id][-5:]

def get_long_term_memory(session_id: str):
    return ". ".join(long_term_memory.get(session_id, []))


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the information from long-term memory if relevant."),
    ("system", "Long-term memory: {long_term_memory}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_chat_history,
    input_messages_key="input",
    history_messages_key="history"
)

def chat(input_text:str, session_id:str):
    long_term_mem = get_long_term_memory(session_id)
    response = chain_with_history.invoke(
        {"input": input_text, "long_term_memory": long_term_mem},
        config={"configurable": {"session_id": session_id}}
    )
    update_long_term_memory(session_id, input_text, response.content)
    return response.content


session_id = "user_123"

print("AI:", chat("Hello! My name is Alice.", session_id))
print("AI:", chat("What's the weather like today?", session_id))
print("AI:", chat("I love sunny days.", session_id))
print("AI:", chat("Do you remember my name?", session_id))

# check memory

print("Conversation History:")
for message in chat_store[session_id].messages:
    print(f"{message.type}: {message.content}")

print("\nLong-term Memory:")
print(get_long_term_memory(session_id))

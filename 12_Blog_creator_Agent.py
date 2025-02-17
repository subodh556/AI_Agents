# Key Components
    # OpenAI's Swarm Package: Facilitates the creation and management of multi-agent interactions.
    # Agents: Include a human admin, AI researcher, content planner, writer, and editor, each with specific responsibilities.
    # Interaction Management: Manages the conversation flow and context among agents.

# Method

# The system follows a structured approach:

# Agent Configuration: Each agent is set up with a specific role and behavior.

# In this step, we define the characteristics and capabilities of each agent. This includes:

    # Setting the agent's name and role
    # Defining the agent's instructions (what it should do)
    # Specifying the functions the agent can call (to interact with other agents or perform specific tasks)

# Role Assignment:

    # Admin: Oversees the project and provides guidance.
    # Researcher: Gathers information on the given topic.
    # Planner: Organizes the research into an outline.
    # Writer: Drafts the blog post based on the outline.
    # Editor: Reviews and edits the draft for quality assurance.
# Each role is crucial for the successful creation of a high-quality blog post. This division of labor allows for specialization and ensures that each aspect of the content creation process receives focused attention.

# Interaction Management: Defines permissible interactions between agents to maintain orderly communication.

# This step involves:

    # Determining which agents can communicate with each other
    # Defining the order of operations (e.g., research before writing)
    # Ensuring that context and information are properly passed between agents

# Task Execution: The admin initiates a task, and agents collaboratively work through researching, planning, writing, and editing.

# The task execution follows a logical flow:

    # Admin sets the topic and initiates the process
    # Planner creates an outline based on the topic
    # Researcher gathers information on each section of the outline
    # Writer uses the research to draft the blog post
    # Editor reviews and refines the final product
# This structured approach ensures a comprehensive and well-researched blog post as the final output.

from dotenv import load_dotenv

load_dotenv()

def complete_blog_post(title,content):
    # create a valid filename from the title
    filename = title.lower().replace(" ","-") + ".md"
    with open(filename, "w", encoding="utf-8") as file:
        file.write(content)
    
    print(f"Blog post '{title}' has been written to {filename}")
    return "Task completed"

from swarm import Agent

def admin_instructions(context_variables):
    topic = context_variables.get("topic", "No topic provided")
    return f"""You are the Admin Agent overseeing the blog post project on the topic: '{topic}'.
Your responsibilities include initiating the project, providing guidance, and reviewing the final content.
Once you've set the topic, call the function to transfer to the planner agent."""


def planner_instructions(context_variables):
    topic = context_variables.get("topic", "No topic provided")
    return f"""You are the Planner Agent. Based on the following topic: '{topic}'
Organize the content into topics and sections with clear headings that will each be individually researched as points in the greater blog post.
Once the outline is ready, call the researcher agent. """


def researcher_instructions(context_variables):
    return """You are the Researcher Agent. your task is to provide dense context and information on the topics outlined by the previous planner agent.
This research will serve as the information that will be formatted into a body of a blog post. Provide comprehensive research like notes for each of the sections outlined by the planner agent.
Once your research is complete, transfer to the writer agent"""


def writer_instructions(context_variables):
    return """You are the Writer Agent. using the prior information write a clear blog post following the outline from the planner agent. 
    Summarise and include as much information relevant from the research into the blog post.
    The blog post should be quite large as the context the context provided should be quite dense.
Write clear, engaging content for each section.
Once the draft is complete, call the function to transfer to the Editor Agent."""


def editor_instructions(context_variables):
    return """You are the Editor Agent. Review and edit th prior blog post completed by the writer agent.
Make necessary corrections and improvements.
Once editing is complete, call the function to complete the blog post"""

def transfer_to_researcher():
    return researcher_agent


def transfer_to_planner():
    return planner_agent


def transfer_to_writer():
    return writer_agent


def transfer_to_editor():
    return editor_agent


def transfer_to_admin():
    return admin_agent


def complete_blog():
    return "Task completed"


admin_agent = Agent(
    name="Admin Agent",
    instructions=admin_instructions,
    functions=[transfer_to_planner],
)

planner_agent = Agent(
    name="Planner Agent",
    instructions=planner_instructions,
    functions=[transfer_to_researcher],
)

researcher_agent = Agent(
    name="Researcher Agent",
    instructions=researcher_instructions,
    functions=[transfer_to_writer],
)

writer_agent = Agent(
    name="Writer Agent",
    instructions=writer_instructions,
    functions=[transfer_to_editor],
)

editor_agent = Agent(
    name="Editor Agent",
    instructions=editor_instructions,
    functions=[complete_blog_post],
)

from swarm.repl import run_demo_loop

def run():
    run_demo_loop(admin_agent,debug=True)

run()
from google.adk.agents import Agent
from .get_account_balance import get_account_balance
from .customer_agent.agent import customer_agent

root_agent = Agent(
    name="banking_agent",
    model="gemini-2.5-flash",
    description="Main banking assistant.",

    instruction="""
    You are the main banking assistant.

    For account balance questions, use get_account_balance.

    For customer profile questions, delegate to customer_agent.

    Do not invent banking or customer data.
    """,

    tools=[get_account_balance],

    sub_agents=[customer_agent],
)
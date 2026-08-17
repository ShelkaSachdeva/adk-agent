from google.adk.agents import Agent
from .get_customer_info import get_customer_info

customer_agent = Agent(
    name="customer_agent",
    model="gemini-2.5-flash",

    description="""
    Handles customer profile requests.
    Use this agent whenever the user asks about a customer ID,
    customer name, segment, city, or customer profile information.
    """,

    instruction="""
    You are a customer information specialist.

    When given a customer ID,
    ALWAYS use the get_customer_info tool.

    Do not invent customer information.
    """,

    tools=[get_customer_info],
)
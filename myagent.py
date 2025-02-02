import os
from phi.agent import Agent
from phi.model.groq import Groq  # Assuming this is how you import Groq Llama
from phi.tools.serpapi_tools import SerpApiTools

def get_travel_plan(groq_api_key, serpapi_key, destination, duration, budget, travel_style):
    if not destination:
        return "Please enter a destination."
    
    os.environ["GROQ_API_KEY"] = groq_api_key
    os.environ["SERP_API_KEY"] = serpapi_key
    
    travel_agent = Agent(
        name="Travel Planner",
        model=Groq(id="llama-3.3-70b-versatile"),
        tools=[SerpApiTools()],
        instructions=[
            "You are a travel planning assistant using Groq Llama.",
            "Help users plan their trips by researching destinations, finding attractions, suggesting accommodations, and providing transportation options.",
            "Give relevant live links of places and hotels by searching on the internet.",
            "Always verify information is current before making recommendations."
        ],
        show_tool_calls=True,
        markdown=True
    )
    
    prompt = f"""Create a comprehensive travel plan for {destination} for {duration} days.

    Travel Preferences:
    - Budget Level: {budget}
    - Travel Styles: {', '.join(travel_style)}

    Please provide a detailed itinerary that includes:
    
    1. Best Time to Visit
    2. Accommodation Recommendations
    3. Day-by-Day Itinerary
    4. Culinary Experiences
    5. Practical Travel Tips
    6. Estimated Total Trip Cost

    Provide sources and relevant links.
    Format the response in a clear markdown format with headings and bullet points.
    """
    
    response = travel_agent.run(prompt)
    
    if hasattr(response, 'content'):
        return response.content.replace('∣', '|').replace('\n\n\n', '\n\n')
    else:
        return str(response)


def ask_travel_question(groq_api_key, serpapi_key, destination, travel_plan, question):
    if not travel_plan:
        return "Please generate a travel plan first."
    
    os.environ["GROQ_API_KEY"] = groq_api_key
    os.environ["SERP_API_KEY"] = serpapi_key
    
    travel_agent = Agent(
        name="Travel Planner",
        model=Groq(id="llama-3.3-70b-versatile"),
        tools=[SerpApiTools()],
        instructions=["Provide focused, concise answers related to the existing travel plan."],
        show_tool_calls=True,
        markdown=True
    )
    
    context_question = f"""
    I have a travel plan for {destination}. Here's the existing plan:
    {travel_plan}

    Now, answer this specific question: {question}
    """
    
    response = travel_agent.run(context_question)
    
    if hasattr(response, 'content'):
        return response.content
    else:
        return str(response)

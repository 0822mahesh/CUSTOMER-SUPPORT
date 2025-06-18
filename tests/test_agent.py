import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.agent_executor import run_support_agent



def test_billing_query():
    query = "I want to know my billing date"
    response = run_support_agent(query)
    assert "billing" in response["output"] and "date" in response["output"]

def test_technical_query():
    query = "Why is the website down?"
    response = run_support_agent(query)
    assert "website" in response["output"] and "down" in response["output"]
   

def test_general_query():
    query = "What are your support hours?"
    response = run_support_agent(query)
    assert "support hours" in response["output"]
   

def test_escalation():
    query = "I'm very unhappy, I need to speak to someone now"
    response = run_support_agent(query)
    assert "escalating" in response["output"] and "agent" in response["output"]
    



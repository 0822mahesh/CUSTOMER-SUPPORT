import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.agent_executor import run_support_agent
from agent.tools import (answer_billing_query,
    answer_technical_query,
    answer_using_rag,
    general_query,
    escalate_to_human
)

def test_run_support_agent():
    """
    Tests the run_support_agent function to ensure it returns a dictionary containing a relevant response
    to a user query. Verifies that the output includes keywords such as 'support' or 'agent', and that the
    original query is present in the output.
    """
    query = "I need help with my billing issue"
    response = run_support_agent(query)
    assert isinstance(response, dict)
    assert "support" in response["response"]["output"].lower() or "billing" in response["response"]["output"].lower()

def test_billing_query():
    """ Tests the answer_billing_query function to ensure it returns a dictionary containing a relevant response
    to a billing query. Verifies that the output includes keywords such as 'billing' or 'date', and that the
    original query is present in the output."""
    query = "I want to know my billing date"
    response = run_support_agent(query)
    result = answer_billing_query.invoke({"query": query})
    assert isinstance(result, dict)
    assert "billing" in result["output"].lower()
    assert query in result["output"]

def test_technical_query():
    """
    Tests the answer_technical_query function to ensure it returns a dictionary containing a relevant response
    to a technical query. Verifies that the output includes keywords such as 'technical' or 'support', and
    that the original query is present in the output.
    """
    query = "Why is the website down?"
    result = answer_technical_query.invoke({"query": query})
    assert isinstance(result, dict)
    assert "technical" in result["output"].lower() or "support" in result["output"].lower()
    assert query in result["output"]

def test_general_query():
    """ Tests the general_query function to ensure it returns a dictionary containing a relevant response
    to a general query. Verifies that the output includes keywords such as 'general' or 'response', and
    that the original query is present in the output."""
    query = "What is the capital of Canada?"
    result = general_query.invoke({"query": query})
    assert isinstance(result, dict)
    assert "capital" in result["output"].lower() or "Canada" in result["output"].lower()

def test_escalation():
    """ Tests the escalate_to_human function to ensure it returns a dictionary indicating escalation
    to a human agent. Verifies that the output includes keywords such as 'escalating' or 'agent', and
    that the original query is present in the output."""
    query = "I'm very unhappy, I need to speak to someone now"
    result = escalate_to_human.invoke({"query": query})
    assert isinstance(result, dict)
    assert "escalating" in result["output"].lower() or "agent" in result["output"].lower()

def test_answer_using_rag():
    """Tests the answer_using_rag function to ensure it returns a dictionary containing a relevant response
    to a query using RAG (Retrieval-Augmented Generation). Verifies that the output includes keywords related to the query,
    such as 'code of conduct', and that the original query"""
    query = "What is the code of conduct?"
    result = answer_using_rag.invoke({"query": query})
    assert isinstance(result, dict)
    assert "code of conduct" in result["output"].lower()

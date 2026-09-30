import networkx as nx

def propagate_risk(graph, confirmed_fraud_nodes):
    """
    F13: Confirm-fraud loop.
    Analyst verdict spreads risk to neighbours using personalized PageRank.
    """
    if not graph or len(graph.nodes) == 0:
        return {}

    # Assign a weight of 1.0 to confirmed fraud accounts, 0.0 to all others
    personalization = {
        node: (1.0 if node in confirmed_fraud_nodes else 0.0) 
        for node in graph.nodes()
    }
    
    # Calculate the boosted scores using incoming link structure
    boosted_scores = nx.pagerank(graph, alpha=0.85, personalization=personalization)
    
    return boosted_scores
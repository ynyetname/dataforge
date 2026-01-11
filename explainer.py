# explainer.py

class Explainer:
    def __init__(self):
        pass
    
    def generate_explanation(self, query, retrieved_chunks, entities, 
                           relationships, graph_path, answer):
        """Generate comprehensive explanation"""
        
        explanation = {
            'answer': answer,
            'reasoning': [],
            'sources': [],
            'entities_used': [],
            'graph_visualization': None
        }
        
        # Document sources
        for i, chunk in enumerate(retrieved_chunks):
            explanation['sources'].append({
                'source_id': i + 1,
                'text_snippet': chunk['chunk'][:200] + "...",
                'relevance_score': chunk['score'],
                'reason': f"This document was retrieved because it has high semantic similarity (score: {chunk['score']:.3f}) to your query."
            })
        
        # Entities used
        explanation['entities_used'] = [
            f"{e['text']} ({e['label']})" for e in entities[:10]
        ]
        
        # Reasoning chain
        explanation['reasoning'] = [
            f"1. Query analyzed and key terms extracted",
            f"2. Retrieved {len(retrieved_chunks)} relevant document chunks",
            f"3. Extracted {len(entities)} entities and {len(relationships)} relationships",
            f"4. Constructed knowledge graph with {len(entities)} nodes",
            f"5. Generated answer using context and entity relationships"
        ]
        
        if graph_path:
            explanation['reasoning'].append(
                f"6. Found reasoning path through graph: {' → '.join(graph_path)}"
            )
        
        return explanation
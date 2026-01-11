import networkx as nx
from pyvis.network import Network

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    """Here entities are acting as nodes"""
    def add_entities(self, entities):
        for entity in entities:
            self.graph.add_node(
                entity['text'],
                label=entity['label'],
                start_entity=entity['start'],
                end_entity = entity['end']
            )
            
    """Here relationships are acting as edges"""
    def add_relationships(self, relationships):
        for rel in relationships:
            self.graph.add_edge(
                rel['subject'],
                rel['object'],
                relation=rel['relation']   
            )     
            
    def visualize(self, output_file='knowledge_graph.html'):
        """interactive visualization"""
        net = Network(height='750px', width='100%', directed=True)
        net.from_nx(self.graph)
        net.show(output_file)
        return output_file
    
    def find_path(self, start_entity, end_entity):
        """Finding path between two entities"""
        try:
            path = nx.shortest_path(self.graph, start_entity, end_entity)
            return path
        except:
            return None
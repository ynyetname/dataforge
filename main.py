# main_system.py
from doc_processor import DocumentProcessor
from retrieval import Retriever
from entity_extractor import EntityExtractor
from knowledge_graph import KnowledgeGraph
from answer_gen import AnswerGenerator
from explainer import Explainer
from sentence_transformers import SentenceTransformer

class ExplainableRAG:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.entity_extractor = EntityExtractor()
        self.kg = KnowledgeGraph()
        self.answer_gen = AnswerGenerator()
        self.explainer = Explainer()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.retriever = None
    
    def ingest_documents(self, file_paths):
        """Loading and processing documents"""
        all_chunks = []
        
        for path in file_paths:
            text = self.doc_processor.load_document(path)
            chunks = self.doc_processor.chunk_document(text)
            all_chunks.extend(chunks)
        
        # Creating embeddings
        embeddings = self.doc_processor.create_embeddings(all_chunks)
        
        # Initializing retriever
        self.retriever = Retriever(embeddings, all_chunks)
        
        # Extracting entities and building knowledge graph
        for chunk in all_chunks:
            entities = self.entity_extractor.extract_entities(chunk)
            relationships = self.entity_extractor.extract_relationships(chunk)
            self.kg.add_entities(entities)
            self.kg.add_relationships(relationships)
        
        print(f"Ingested {len(all_chunks)} chunks from {len(file_paths)} documents")
    
    def query(self, user_query):
        """Processing query and returning explainable answer"""
        
        query_embedding = self.embedding_model.encode(user_query)
        
        retrieved_chunks = self.retriever.search(query_embedding, top_k=5)
        
        query_entities = self.entity_extractor.extract_entities(user_query)
        
        context_text = " ".join([chunk['chunk'] for chunk in retrieved_chunks])
        context_entities = self.entity_extractor.extract_entities(context_text)
        context_relationships = self.entity_extractor.extract_relationships(context_text)
        
        graph_path = None
        if len(query_entities) >= 2 and len(context_entities) > 0:
            try:
                graph_path = self.kg.find_path(
                    query_entities[0]['text'],
                    context_entities[0]['text']
                )
            except:
                pass
        
        answer = self.answer_gen.generate_answer(
            user_query,
            retrieved_chunks,
            context_entities,
            graph_path
        )
        
        explanation = self.explainer.generate_explanation(
            user_query,
            retrieved_chunks,
            context_entities,
            context_relationships,
            graph_path,
            answer
        )
        
        graph_file = self.kg.visualize()
        explanation['graph_visualization'] = graph_file
        
        return explanation

if __name__ == "__main__":
    system = ExplainableRAG()
    
    system.ingest_documents(['document1.pdf', 'document2.pdf'])
    
    result = system.query("What is the impact of climate change on agriculture?")
    
    print("ANSWER:", result['answer'])
    print("\nREASONING:")
    for step in result['reasoning']:
        print(f"  {step}")
    print("\nSOURCES:")
    for source in result['sources']:
        print(f"  - {source['text_snippet']}")
    
    print("\nENTITIES USED:")
    for entity in result['entities_used']:
        print(f"  - {entity}")
    
    print("\nGRAPH VISUALIZATION:")
    print(f"  {result['graph_visualization']}")
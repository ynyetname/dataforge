import spacy

class EntityExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
    
    def extract_entities(self, text):
        """Extracting named entities from text"""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        return entities
    
    def extract_relationships(self, text):
        """Extracting relationships between entities"""
        doc = self.nlp(text)
        relationships = []
        
        for token in doc:
            if token.dep_ in ['nsubj', 'dobj']:
                subject = token.text
                verb = token.head.text
                for child in token.head.children:
                    if child.dep_ == 'dobj':
                        obj = child.text
                        relationships.append({
                            'subject': subject,
                            'relation': verb,
                            'object': obj
                        })
        return relationships
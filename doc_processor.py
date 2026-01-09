import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np

class DocProcessor:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.docs = []
        self.embeddings = []
        
    def load_docs(self, file_path):
        """Loading and Parsing the document"""
        text = ""
        if file_path.endswith('.pdf'):
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfFileReader(f)
                for page in reader.pages:
                    text += page.extract_text()               
        return text
    
    def chunk_doc(self, text, chunk_size=500, overlap=50):  
        """Splitting documents into chunks"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
    
    def create_embeddings(self, chunks):
        """Creating vector embeddings for each chunk"""
        embeddings = self.model.encode(chunks)
        return embeddings
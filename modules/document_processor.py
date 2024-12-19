# Remplacer l'ancienne importation par :
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredPDFLoader as PDFLoader
from langchain_community.document_loaders import Docx2txtLoader as DocxLoader

class DocumentProcessor:
    def __init__(self, upload_dir):
        self.upload_dir = upload_dir

    def process_uploaded_document(self, file_path):
        file_extension = file_path.split('.')[-1].lower()
        
        try:
            if file_extension == 'txt':
                loader = TextLoader(file_path)
            elif file_extension == 'pdf':
                loader = PDFLoader(file_path)
            elif file_extension in ['docx', 'doc']:
                loader = DocxLoader(file_path)
            else:
                raise ValueError(f"Format de fichier non supporté: {file_extension}")
                
            documents = loader.load()
            return [doc.page_content for doc in documents]
            
        except Exception as e:
            print(f"Erreur lors du traitement du document: {str(e)}")
            return []

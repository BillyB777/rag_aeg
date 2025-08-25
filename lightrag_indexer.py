#!/usr/bin/env python3
"""
LightRAG Document Indexer
=========================

This script indexes PDFs and documents using LightRAG with OpenAI.
Run this ONCE to create the knowledge graph and embeddings.
Then use the MCP server to query without additional API calls.
"""

import os
import asyncio
import glob
import argparse
from pathlib import Path
import pdfplumber
from docx import Document
from dotenv import load_dotenv
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger

# Load environment variables
load_dotenv()

# Setup logging
setup_logger("lightrag", level="INFO")

class LightRAGIndexer:
    def __init__(self, 
                 documents_dir="./documents", 
                 working_dir="./lightrag_storage"):
        self.documents_dir = documents_dir
        self.working_dir = working_dir
        self.rag = None
        
        # Create working directory if it doesn't exist
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)
    
    async def initialize_rag(self):
        """Initialize LightRAG with OpenAI models."""
        print("🚀 Initializing LightRAG with OpenAI models...")
        
        self.rag = LightRAG(
            working_dir=self.working_dir,
            embedding_func=openai_embed,
            llm_model_func=gpt_4o_mini_complete,
        )
        
        # IMPORTANT: Both initialization calls are required!
        await self.rag.initialize_storages()  # Initialize storage backends
        await initialize_pipeline_status()  # Initialize processing pipeline
        
        print("✅ LightRAG initialized successfully")
    
    def find_document_files(self):
        """Find all PDF and DOCX files in the documents directory."""
        pdf_pattern = os.path.join(self.documents_dir, "**", "*.pdf")
        docx_pattern = os.path.join(self.documents_dir, "**", "*.docx")
        
        pdf_files = glob.glob(pdf_pattern, recursive=True)
        docx_files = glob.glob(docx_pattern, recursive=True)
        
        all_files = pdf_files + docx_files
        
        # Filter out temporary files (Word creates ~$filename.docx temp files)
        filtered_files = []
        for file_path in all_files:
            filename = os.path.basename(file_path)
            if not filename.startswith('~$'):
                filtered_files.append(file_path)
        
        return sorted(filtered_files)  # Sort for consistent processing order
    
    def extract_pdf_text(self, pdf_path):
        """Extract text from a PDF file."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        pages.append(text)
                
                full_text = "\n".join(pages)
                return full_text if full_text.strip() else None
                
        except Exception as e:
            print(f"Error extracting PDF {pdf_path}: {e}")
            return None
    
    def extract_docx_text(self, docx_path):
        """Extract text from a DOCX file."""
        try:
            doc = Document(docx_path)
            paragraphs = []
            
            # Extract all paragraphs
            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if text:
                    paragraphs.append(text)
            
            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        cell_text = cell.text.strip()
                        if cell_text:
                            row_text.append(cell_text)
                    if row_text:
                        paragraphs.append(" | ".join(row_text))
            
            full_text = "\n".join(paragraphs)
            return full_text if full_text.strip() else None
                
        except Exception as e:
            print(f"Error extracting DOCX {docx_path}: {e}")
            return None
    
    def extract_document_text(self, file_path):
        """Extract text from either PDF or DOCX file."""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self.extract_pdf_text(file_path)
        elif file_ext == '.docx':
            return self.extract_docx_text(file_path)
        else:
            print(f"Unsupported file type: {file_ext}")
            return None
    
    async def index_documents(self):
        """Index all PDF and DOCX documents using LightRAG."""
        print("📁 Finding document files...")
        document_files = self.find_document_files()
        
        if not document_files:
            print(f"❌ No PDF or DOCX files found in {self.documents_dir}")
            return False
        
        # Count file types
        pdf_count = len([f for f in document_files if f.lower().endswith('.pdf')])
        docx_count = len([f for f in document_files if f.lower().endswith('.docx')])
        
        print(f"📄 Found {len(document_files)} documents to index:")
        print(f"   📕 {pdf_count} PDF files")
        print(f"   📘 {docx_count} DOCX files")
        
        # Process each document
        successful = 0
        failed = 0
        
        for i, doc_path in enumerate(document_files, 1):
            filename = os.path.basename(doc_path)
            relative_path = os.path.relpath(doc_path, self.documents_dir)
            file_ext = os.path.splitext(filename)[1].upper()
            
            print(f"\n📄 Processing {i}/{len(document_files)}: {filename} ({file_ext})")
            
            # Extract text
            text = self.extract_document_text(doc_path)
            
            if not text:
                print(f"   ⚠️  Skipped (no extractable text)")
                failed += 1
                continue
            
            # Prepare document with metadata
            document_text = f"""Document: {filename}
Path: {relative_path}
Type: Project Document

Content:
{text}"""
            
            try:
                # Insert into LightRAG (this uses OpenAI for embeddings and knowledge graph)
                print(f"   🔄 Indexing with LightRAG...")
                await self.rag.ainsert(document_text)
                print(f"   ✅ Successfully indexed ({len(text.split())} words)")
                successful += 1
                
                # Small delay to respect rate limits
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"   ❌ Failed to index: {e}")
                failed += 1
                
                # If rate limited, wait longer
                if "rate" in str(e).lower() or "limit" in str(e).lower():
                    print("   ⏳ Rate limited, waiting 30 seconds...")
                    await asyncio.sleep(30)
        
        print(f"\n🎉 Indexing Complete!")
        print(f"   ✅ Successfully indexed: {successful} documents")
        print(f"   ❌ Failed: {failed} documents")
        print(f"   💾 LightRAG storage: {self.working_dir}")
        
        return successful > 0
    
    async def test_queries(self):
        """Test the indexed knowledge base with sample queries."""
        print(f"\n🔍 Testing LightRAG Knowledge Base")
        print("=" * 40)
        
        test_queries = [
            "What is the main project about?",
            "Who are the project partners?", 
            "What are the main objectives?",
            "What is the project budget and timeline?",
            "What are the key deliverables?"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            print("-" * 30)
            
            try:
                # Test hybrid mode
                result = await self.rag.aquery(
                    query,
                    param=QueryParam(mode="hybrid", response_type="Single Paragraph")
                )
                
                print(f"✅ Response: {result[:200]}...")
                
                # Small delay between queries
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"❌ Query failed: {e}")
    
    async def finalize(self):
        """Clean up resources."""
        if self.rag:
            await self.rag.finalize_storages()

async def main():
    """Main indexing process."""
    parser = argparse.ArgumentParser(description="Index documents with LightRAG")
    parser.add_argument(
        "documents_dir",
        nargs="?",
        default=os.getenv("DOCUMENTS_DIR", "./documents"),
        help="Directory containing PDF documents to index"
    )
    parser.add_argument(
        "--working-dir",
        default=os.getenv("LIGHTRAG_STORAGE_DIR", "./lightrag_storage"),
        help="Directory to store LightRAG data"
    )
    
    args = parser.parse_args()
    
    print("LightRAG Document Indexer")
    print("=" * 50)
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set.")
        print("Please set it in .env file or environment")
        return
    
    print("✅ OpenAI API key found")
    print(f"📁 Documents directory: {args.documents_dir}")
    print(f"💾 Storage directory: {args.working_dir}")
    
    # Check if documents directory exists
    if not os.path.exists(args.documents_dir):
        print(f"❌ Documents directory not found: {args.documents_dir}")
        print("Please create the directory and add PDF files to index")
        return
    
    indexer = LightRAGIndexer(
        documents_dir=args.documents_dir,
        working_dir=args.working_dir
    )
    
    try:
        # Initialize LightRAG
        await indexer.initialize_rag()
        
        # Index all documents
        success = await indexer.index_documents()
        
        if success:
            # Test the knowledge base
            await indexer.test_queries()
            
            print(f"\n🎯 Next Steps:")
            print(f"1. LightRAG knowledge base created at: {indexer.working_dir}")
            print(f"2. Configure Claude Desktop with the MCP server")
            print(f"3. Query via Claude Desktop (no more OpenAI API calls needed!)")
        else:
            print(f"\n❌ Indexing failed. Please check the errors above.")
    
    except Exception as e:
        print(f"❌ Fatal error: {e}")
    
    finally:
        await indexer.finalize()

if __name__ == "__main__":
    asyncio.run(main())
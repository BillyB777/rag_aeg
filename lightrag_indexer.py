#!/usr/bin/env python3
"""
LightRAG CoEvolution Indexer
============================

This script indexes all CoEvolution PDFs using LightRAG with OpenAI.
Run this ONCE to create the knowledge graph and embeddings.
Then use the MCP server to query without additional API calls.
"""

import os
import asyncio
import glob
from pathlib import Path
import pdfplumber
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger

# Setup logging
setup_logger("lightrag", level="INFO")

class CoEvolutionLightRAGIndexer:
    def __init__(self, 
                 coevolution_dir="./COEVOLUTION", 
                 working_dir="./lightrag_storage"):
        self.coevolution_dir = coevolution_dir
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
    
    def find_pdf_files(self):
        """Find all PDF files in the COEVOLUTION directory."""
        pdf_pattern = os.path.join(self.coevolution_dir, "**", "*.pdf")
        pdf_files = glob.glob(pdf_pattern, recursive=True)
        return pdf_files
    
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
            print(f"Error extracting {pdf_path}: {e}")
            return None
    
    async def index_documents(self):
        """Index all PDF documents using LightRAG."""
        print("📁 Finding PDF files...")
        pdf_files = self.find_pdf_files()
        
        if not pdf_files:
            print(f"❌ No PDF files found in {self.coevolution_dir}")
            return False
        
        print(f"📄 Found {len(pdf_files)} PDF files to index")
        
        # Process each PDF
        successful = 0
        failed = 0
        
        for i, pdf_path in enumerate(pdf_files, 1):
            filename = os.path.basename(pdf_path)
            relative_path = os.path.relpath(pdf_path, self.coevolution_dir)
            
            print(f"\n📄 Processing {i}/{len(pdf_files)}: {filename}")
            
            # Extract text
            text = self.extract_pdf_text(pdf_path)
            
            if not text:
                print(f"   ⚠️  Skipped (no extractable text)")
                failed += 1
                continue
            
            # Prepare document with metadata
            document_text = f"""Document: {filename}
Path: {relative_path}
Type: CoEvolution Project Document

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
            "What is the CoEvolution project about?",
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
    print("LightRAG CoEvolution Indexer")
    print("=" * 50)
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set.")
        print("Please set it: export OPENAI_API_KEY='your-key-here'")
        return
    
    print("✅ OpenAI API key found")
    
    indexer = CoEvolutionLightRAGIndexer()
    
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
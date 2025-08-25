# LightRAG Document Indexer

A simple document indexing and querying system using LightRAG with OpenAI for proposals, grants, and other project documents.

## Setup

1. **Install Dependencies**
   ```bash
   pip install python-dotenv lightrag-hku pdfplumber python-docx mcp
   ```

2. **Configure Environment**
   Copy `.env.example` to `.env` and add your OpenAI API key:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env`:
   ```
   OPENAI_API_KEY=sk-your-actual-openai-api-key-here
   LIGHTRAG_STORAGE_DIR=./lightrag_storage
   DOCUMENTS_DIR=./documents
   ```

3. **Prepare Documents**
   Create a directory and add PDF/DOCX files:
   ```bash
   mkdir documents
   # Copy your PDF and DOCX files to documents/
   ```

## Usage

### Index Documents
Index all PDFs and DOCX files in a directory:
```bash
python3 lightrag_indexer.py ./documents
```

Or use default directory from .env:
```bash 
python3 lightrag_indexer.py
```

Or specify custom storage location:
```bash
python3 lightrag_indexer.py ./my-docs --working-dir ./custom-storage
```

### Query via MCP Server
After indexing, use the MCP server with Claude Desktop:
```bash
python3 lightrag_mcp_final.py
```

## Files

- `lightrag_indexer.py` - Index PDF and DOCX documents into LightRAG knowledge base
- `lightrag_mcp_final.py` - MCP server for querying the knowledge base
- `.env.example` - Environment variable template
- `.gitignore` - Git ignore file (includes .env)

## Examples

**Index proposal documents:**
```bash
python3 lightrag_indexer.py ./proposals
```

**Index grant applications:**
```bash
python3 lightrag_indexer.py ./grants --working-dir ./grants_storage
```

**Index multiple document types:**
```bash
python3 lightrag_indexer.py ./all-documents
```

The indexer will:
1. Find all PDF and DOCX files recursively  
2. Extract text content (including tables from DOCX)
3. Create embeddings and knowledge graph with OpenAI
4. Store in LightRAG format for fast querying

After indexing, the MCP server allows querying without additional OpenAI API calls.

## Claude Desktop Configuration

Add to your Claude Desktop config:

```json
{
  "mcpServers": {
    "lightrag": {
      "command": "python3",
      "args": ["path/to/lightrag_mcp_final.py"],
      "env": {
        "OPENAI_API_KEY": "your-key-here"
      }
    }
  }
}
```
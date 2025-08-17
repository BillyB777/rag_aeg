# LightRAG CoEvolution Knowledge Base

A powerful document search and knowledge extraction system built with LightRAG for the CoEvolution project.

## Overview

This system indexes CoEvolution project documents using LightRAG (knowledge graph + vector embeddings) and provides an MCP server interface for Claude Desktop integration.

## Project Structure

```
├── lightrag_indexer.py          # Main indexing script
├── lightrag_mcp_final.py        # MCP server for Claude Desktop
├── LightRAG/                    # Core LightRAG library
├── COEVOLUTION/                 # Source PDF documents
├── lightrag_storage/           # Generated knowledge base (created after indexing)
└── README.md                   # This file
```

## Requirements

- Python 3.8+
- OpenAI API key
- Required packages (install via pip):
  - lightrag-hku
  - pdfplumber
  - mcp

## Setup Instructions

### 1. Install Dependencies

```bash
pip install lightrag-hku pdfplumber mcp
```

### 2. Set OpenAI API Key

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

### 3. Index Documents (One-time setup)

This step processes all PDF documents and creates the knowledge base:

```bash
python3 lightrag_indexer.py
```

The indexer will:
- Find all PDF files in the `COEVOLUTION/` directory
- Extract text content
- Create knowledge graph and embeddings using OpenAI
- Store everything in `lightrag_storage/`

**Note**: This step requires OpenAI API calls and may take 10-30 minutes depending on document count.

### 4. Configure Claude Desktop (Optional)

To use with Claude Desktop, add this to your Claude Desktop config:

```json
{
  "mcpServers": {
    "lightrag-coevolution": {
      "command": "python3",
      "args": ["path/to/lightrag_mcp_final.py"],
      "env": {
        "OPENAI_API_KEY": "your-key-here"
      }
    }
  }
}
```

## Usage

### Direct Python Usage

```python
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

# Initialize LightRAG
rag = LightRAG(
    working_dir="./lightrag_storage",
    embedding_func=openai_embed,
    llm_model_func=gpt_4o_mini_complete,
)

# Query the knowledge base
result = await rag.aquery(
    "What is the CoEvolution project about?",
    param=QueryParam(mode="hybrid")
)
print(result)
```

### Claude Desktop Integration

Once configured, use these tools in Claude Desktop:

- **retrieval**: Search the knowledge base
  - `query`: Your question about CoEvolution
  - `mode`: "local", "global", "hybrid", or "naive"
  - `use_generation`: True for AI-generated responses, False for context only

- **storage_status**: Check knowledge base status

### Example Queries

- "What are the main objectives of the CoEvolution project?"
- "Who are the project partners and their roles?"
- "What is the project timeline and key deliverables?"
- "What are the technical requirements for CoEvolution?"
- "Summarize the grant agreement details"

## Query Modes

- **local**: Focused search within specific document sections
- **global**: Broad search across the entire knowledge graph
- **hybrid**: Combination of local and global (recommended)
- **naive**: Simple keyword-based search

## Storage Details

After indexing, the `lightrag_storage/` directory contains:

- `kv_store_*.json`: Key-value stores for documents, entities, relations
- `vdb_*.json`: Vector embeddings for semantic search
- `graph_*.graphml`: Knowledge graph structure
- Cache files for performance optimization

## Troubleshooting

### "No storage found" error
Run the indexing step: `python3 lightrag_indexer.py`

### OpenAI API errors
- Check your API key is set correctly
- Verify you have sufficient API credits
- Rate limiting may cause delays during indexing

### Empty results
- Ensure PDFs contain extractable text
- Try different query modes
- Check that indexing completed successfully

## Performance Notes

- **First-time indexing**: Requires OpenAI API calls (~$5-15 depending on document size)
- **Querying**: 
  - Context-only mode: No additional API calls
  - Generation mode: Uses OpenAI for response generation
- **Storage size**: Typically 10-50MB for moderate document collections

## License

This project uses the LightRAG library and follows its licensing terms.

## Support

For issues with:
- LightRAG core functionality: Check the [LightRAG repository](https://github.com/HKUDS/LightRAG)
- CoEvolution-specific setup: Review this README and configuration files
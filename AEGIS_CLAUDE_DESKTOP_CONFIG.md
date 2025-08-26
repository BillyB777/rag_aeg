# LightRAG Unified MCP - Claude Desktop Configuration

## Claude Desktop Configuration

**Single MCP server** providing both basic retrieval and AEGIS proposal writing tools.

Add this to your Claude Desktop MCP configuration file:

```json
{
  "mcpServers": {
    "lightrag-unified": {
      "command": "wsl",
      "args": [
        "-e", "bash", "-c",
        "cd /mnt/c/Users/Administrator/Documents/LightRag && python3 lightrag_unified_mcp.py"
      ],
      "env": {
        "OPENAI_API_KEY": "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"
      }
    }
  }
}
```

## Available Tools

### Basic LightRAG Tools

#### 1. `retrieval`
Simple search through your indexed documents
- **Query modes**: local, global, hybrid, naive
- **Options**: Context-only or generated responses
- **Usage**: General document search and exploration

#### 2. `storage_status`
Check LightRAG storage health and file inventory
- Shows indexed files and storage size
- Verifies knowledge base is ready

### AEGIS Proposal Writing Tools

#### 3. `write_text`
Generate or revise EU Horizon text with LLM+RAG
- **Modes**: draft, revise
- **Doc types**: proposal, deliverable, concept_note  
- **Features**: Auto-citation, coverage analysis, word budget control

#### 4. `assess_tone`
LLM-only style assessment against EU document profiles
- **Profiles**: eu_proposal, eu_deliverable, concept_note, custom
- **Metrics**: clarity, formality, vagueness, conciseness
- **Output**: Scores + edit instructions

#### 5. `tech_assess`
Technical validation with LLM+RAG against AEGIS assets
- **Features**: TRL assessment, asset matching, gap analysis
- **Focus**: AEGIS capabilities, pilot validation, overpromise detection

## Example Workflows

### Basic Document Search
```
1. retrieval(query="project partners", mode="hybrid")
2. storage_status()  [check what's indexed]
```

### Draft a Proposal Section
```
1. write_text(mode="draft", doc_type="proposal", section_id="Impact.1", ...)
2. assess_tone(profile="eu_proposal", ...)  [optional polish]
3. write_text(mode="revise", revise={...})  [if needed]
```

### Technical Validation Workflow
```
1. tech_assess(text="...", trl_target=6, ...)
2. write_text(mode="revise", revise={gaps/overpromises})  [if needed]
```

### Combined Research + Writing
```
1. retrieval(query="AEGIS tools capabilities", mode="global")
2. write_text(mode="draft", prompt="Based on retrieved context...")
3. tech_assess(text="draft", trl_target=6)
4. assess_tone(text="draft", profile="eu_deliverable")
```

## System Prompt for Claude Code

Add this to your project instructions:

```
You are the AEGIS drafting assistant. Use **only three MCP tools**: `write_text`, `assess_tone`, `tech_assess`. Do **not** implement hard rules; reason with the LLM and consult LightRAG inside each tool call using the provided retrieval hints. For any drafting task:

1. Call `write_text` with user ToC/must‑cover and the appropriate `doc_type`.
2. If the user wants stylistic polish or you detect mismatches, call `assess_tone` and then `write_text(mode=revise)` with its `edit_instructions`.
3. If the text asserts AEGIS capabilities, pilots, datasets, or TRL progress, call `tech_assess`; if it finds gaps/overpromises, revise via `write_text`.
4. Return the **final text** plus the **citations** and **assessment summaries**.

Keep outputs concise and structured. Prefer minimal tool calls. All retrieval is handled *inside* the tools—you only pass retrieval hints.
```

## Benefits of Unified Server

- **Single MCP connection** - easier configuration and management
- **Seamless workflow** - combine basic search with advanced AEGIS tools
- **Consistent storage** - all tools use the same indexed documents
- **Reduced overhead** - one server process instead of multiple

## Notes

- **Restart Claude Desktop** after updating the configuration
- **Remove old MCP configs** - delete separate lightrag/aegis entries
- Ensure your `.env` file has the correct `OPENAI_API_KEY`
- The server uses existing LightRAG storage from document indexing
- **5 total tools available**: retrieval, storage_status, write_text, assess_tone, tech_assess
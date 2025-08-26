#!/usr/bin/env python3
"""
LightRAG Unified MCP Server
===========================

Combined MCP server providing:
1. Basic retrieval (from lightrag_mcp_final.py)
2. Storage status checking
3. AEGIS Minimal tools (write_text, assess_tone, tech_assess)

Single MCP server for all LightRAG functionality.
"""

import asyncio
import json
import logging
import sys
import os
from typing import List, Optional, Dict, Any
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file.")

# Import LightRAG components
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lightrag-unified-mcp")

# LLM Prompt Templates for AEGIS tools
WRITE_TEXT_SYSTEM = """You are an AEGIS proposal writer. Draft or revise text for EU Horizon documents (proposals, deliverables, concept notes) using only the provided retrieval context. You must:
- Cover all requested bullets concisely within the word budget.
- Ground key claims in the retrieved evidence; add inline anchors like [R1], [R2].
- Prefer precise commitments and concrete KPIs over vague promises.
- If required evidence is missing, surface it in `open_questions` instead of guessing.
- Output STRICT JSON only."""

ASSESS_TONE_SYSTEM = """You are an AEGIS senior editor. Assess style and tone of EU Horizon text purely by expert judgement (no deterministic metrics). Score against the requested profile and provide actionable edit instructions. Output STRICT JSON only."""

TECH_ASSESS_SYSTEM = """You are an AEGIS technical reviewer. Validate claims against AEGIS assets and TRL progression using only retrieved context. Identify matches, gaps, overpromises, and required evidence to reach the target TRL. Prefer conservative, evidence-backed judgements. Output STRICT JSON only."""

class LightRAGUnifiedEngine:
    def __init__(self, working_dir: str = None):
        self.working_dir = working_dir or os.getenv("LIGHTRAG_STORAGE_DIR", "./lightrag_storage")
        self.rag = None
        self.initialized = False
        
        # Check if LightRAG storage exists
        if not os.path.exists(self.working_dir):
            logger.error(f"LightRAG storage not found: {self.working_dir}")
    
    async def initialize(self):
        """Initialize LightRAG instance with required functions."""
        if self.initialized:
            return
        
        try:
            # Create LightRAG instance with the same functions used during indexing
            self.rag = LightRAG(
                working_dir=self.working_dir,
                embedding_func=openai_embed,
                llm_model_func=gpt_4o_mini_complete,
            )
            
            # Initialize storage backends and pipeline
            await self.rag.initialize_storages()
            await initialize_pipeline_status()
            
            self.initialized = True
            logger.info("LightRAG Unified Engine initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize LightRAG: {e}")
            raise
    
    async def query_context_only(self, query: str, mode: str = "local") -> str:
        """Query using context-only mode (no LLM generation)."""
        if not self.initialized:
            await self.initialize()
        
        try:
            result = await self.rag.aquery(
                query,
                param=QueryParam(
                    mode=mode,
                    only_need_context=True,  # Only retrieve context, no generation
                    top_k=20,
                    chunk_top_k=8,
                    max_total_tokens=12000
                )
            )
            
            return result if result else "No relevant context found in the knowledge base."
            
        except Exception as e:
            logger.error(f"Context query error: {e}")
            return f"Query error: {str(e)}"
    
    async def query_with_generation(self, query: str, mode: str = "hybrid") -> str:
        """Query with LLM generation (uses OpenAI)."""
        if not self.initialized:
            await self.initialize()
        
        try:
            result = await self.rag.aquery(
                query,
                param=QueryParam(
                    mode=mode,
                    response_type="Multiple Paragraphs",
                    top_k=15,
                    chunk_top_k=5,
                    max_total_tokens=15000
                )
            )
            
            return result if result else "No response generated."
            
        except Exception as e:
            logger.error(f"Generation query error: {e}")
            return f"Query error: {str(e)}"
    
    async def get_storage_info(self) -> dict:
        """Get storage status information."""
        try:
            storage_info = {
                "working_dir": self.working_dir,
                "exists": os.path.exists(self.working_dir),
                "files": []
            }
            
            if os.path.exists(self.working_dir):
                for item in os.listdir(self.working_dir):
                    item_path = os.path.join(self.working_dir, item)
                    if os.path.isfile(item_path):
                        storage_info["files"].append({
                            "name": item,
                            "size_mb": round(os.path.getsize(item_path) / (1024 * 1024), 2)
                        })
            
            return storage_info
            
        except Exception as e:
            logger.error(f"Storage info error: {e}")
            return {"error": str(e)}
    
    # AEGIS Tool Methods
    def format_retrieval_passages(self, raw_result: str, max_passages: int = 10) -> List[Dict[str, str]]:
        """Parse raw LightRAG result into structured passages for AEGIS tools."""
        passages = []
        
        if not raw_result or "No relevant context found" in raw_result:
            return passages
        
        # Split into chunks and create passage objects
        chunks = raw_result.split('\n\n')
        
        for i, chunk in enumerate(chunks[:max_passages]):
            if chunk.strip():
                passages.append({
                    "id": f"R{i+1}",
                    "doc_class": "documents",  # Simplified since we use unified storage
                    "source_uri": f"/documents/chunk_{i+1}",
                    "snippet": chunk.strip()[:500] + ("..." if len(chunk) > 500 else "")
                })
        
        return passages
    
    async def llm_call(self, system_prompt: str, user_prompt: str) -> str:
        """Make direct LLM call for AEGIS tools."""
        try:
            # Format prompt for gpt_4o_mini_complete function
            full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"
            
            response = await gpt_4o_mini_complete(full_prompt, **{})
            return response
            
        except Exception as e:
            logger.error(f"LLM call error: {e}")
            return json.dumps({"error": f"LLM call failed: {str(e)}"})

# Global engine instance
query_engine = LightRAGUnifiedEngine()

# MCP Server
server = Server("lightrag-unified-mcp")

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """List all unified tools: basic retrieval + AEGIS tools."""
    try:
        return [
            # Basic retrieval tool (from lightrag_mcp_final.py)
            types.Tool(
                name="retrieval",
                description="Search the project knowledge base using LightRAG. Returns relevant context from indexed documents.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Question or search terms about the project documents"
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["local", "global", "hybrid", "naive"],
                            "description": "Search mode: 'local' for focused search, 'hybrid' for comprehensive, 'global' for broad context",
                            "default": "hybrid"
                        },
                        "use_generation": {
                            "type": "boolean",
                            "description": "Whether to use LLM generation (requires OpenAI) or just return context",
                            "default": False
                        }
                    },
                    "required": ["query"]
                }
            ),
            
            # Storage status tool
            types.Tool(
                name="storage_status",
                description="Check the status of the LightRAG storage and indexed documents",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            ),
            
            # AEGIS write_text tool
            types.Tool(
                name="write_text",
                description="Generate or revise text with in-tool RAG over AEGIS corpora.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["draft", "revise"],
                            "description": "Draft new text or revise existing text"
                        },
                        "doc_type": {
                            "type": "string", 
                            "enum": ["proposal", "deliverable", "concept_note"],
                            "description": "Type of document being written"
                        },
                        "section_id": {
                            "type": "string",
                            "description": "Section identifier (e.g., 'Impact.1')"
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Writing instructions"
                        },
                        "must_cover": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Required topics to cover"
                        },
                        "objectives": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Project objectives"
                        },
                        "kpis": {
                            "type": "array", 
                            "items": {"type": "string"},
                            "description": "Key performance indicators"
                        },
                        "word_budget": {
                            "type": "integer",
                            "default": 800,
                            "description": "Target word count"
                        },
                        "tone": {
                            "type": "object",
                            "properties": {
                                "audience": {"type": "string"},
                                "formality": {"type": "string"},
                                "voice": {"type": "string"}
                            },
                            "description": "Tone specifications"
                        },
                        "revise": {
                            "type": "object",
                            "properties": {
                                "previous_text": {"type": "string"},
                                "instructions": {
                                    "type": "array",
                                    "items": {"type": "object"}
                                }
                            },
                            "description": "Revision parameters (required for mode=revise)"
                        },
                        "retrieval": {
                            "type": "object",
                            "properties": {
                                "project_id": {"type": "string"},
                                "filters": {"type": "object"},
                                "top_k": {"type": "integer", "default": 15},
                                "chunk_top_k": {"type": "integer", "default": 6}
                            },
                            "description": "Retrieval hints"
                        },
                        "citations_mode": {
                            "type": "string",
                            "enum": ["inline_ids", "footnotes", "none"],
                            "default": "inline_ids",
                            "description": "Citation format"
                        }
                    },
                    "required": ["mode", "doc_type", "prompt"]
                }
            ),
            
            # AEGIS assess_tone tool
            types.Tool(
                name="assess_tone",
                description="LLM-only style assessment against a profile or custom rubric.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to assess"
                        },
                        "profile": {
                            "type": "string",
                            "enum": ["eu_proposal", "eu_deliverable", "concept_note", "custom"],
                            "description": "Style profile to assess against"
                        },
                        "targets": {
                            "type": "object",
                            "properties": {
                                "clarity": {"type": "array", "items": {"type": "number"}},
                                "formality": {"type": "array", "items": {"type": "number"}},
                                "vagueness": {"type": "array", "items": {"type": "number"}},
                                "conciseness": {"type": "array", "items": {"type": "number"}}
                            },
                            "description": "Target score ranges [min, max]"
                        },
                        "custom_rubric": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Custom style rules"
                        }
                    },
                    "required": ["text", "profile"]
                }
            ),
            
            # AEGIS tech_assess tool
            types.Tool(
                name="tech_assess",
                description="LLM+RAG technical validation vs AEGIS assets and TRL.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to validate technically"
                        },
                        "project_id": {
                            "type": "string",
                            "description": "Project identifier"
                        },
                        "trl_target": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 9,
                            "description": "Target Technology Readiness Level"
                        },
                        "retrieval": {
                            "type": "object",
                            "properties": {
                                "filters": {"type": "object"},
                                "top_k": {"type": "integer", "default": 20}
                            },
                            "description": "Technical retrieval parameters"
                        }
                    },
                    "required": ["text"]
                }
            )
        ]
    except Exception:
        logger.exception("list_tools failed")
        return []

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> List[types.TextContent]:
    """Handle all tool calls: basic + AEGIS."""
    try:
        await query_engine.initialize()
        
        if name == "retrieval":
            return await handle_retrieval(arguments)
        elif name == "storage_status":
            return await handle_storage_status(arguments)
        elif name == "write_text":
            return await handle_write_text(arguments)
        elif name == "assess_tone":
            return await handle_assess_tone(arguments)
        elif name == "tech_assess":
            return await handle_tech_assess(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
            
    except Exception as e:
        logger.exception(f"Tool {name} failed")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]

# Basic retrieval handlers (from lightrag_mcp_final.py)
async def handle_retrieval(arguments: dict) -> List[types.TextContent]:
    """Handle retrieval using simple approach that matches lightrag_mcp_final.py."""
    query = arguments["query"]
    mode = arguments.get("mode", "hybrid") 
    use_generation = arguments.get("use_generation", False)
    
    # Use simple query method that matches the working approach
    if use_generation:
        result = await query_engine.query_with_generation(query, mode)
        response_type = "Generated Response"
    else:
        result = await query_engine.query_context_only(query, mode)
        response_type = "Context Only"
    
    # Format response similar to lightrag_mcp_final.py
    response_text = f"**Project LightRAG Search**\n\n"
    response_text += f"**Query**: {query}\n"
    response_text += f"**Mode**: {mode}\n" 
    response_text += f"**Type**: {response_type}\n\n"
    response_text += f"**Result**:\n{result}"
    
    return [types.TextContent(type="text", text=response_text)]

async def handle_storage_status(arguments: dict) -> List[types.TextContent]:
    """Handle storage status check."""
    try:
        info = await query_engine.get_storage_info()
        
        if "error" in info:
            response = f"**Storage Status Error**\n\n{info['error']}"
        else:
            response = f"**LightRAG Storage Status**\n\n"
            response += f"**Directory**: {info['working_dir']}\n"
            response += f"**Exists**: {'✅ Yes' if info['exists'] else '❌ No'}\n\n"
            
            if info['files']:
                response += f"**Storage Files** ({len(info['files'])} files):\n"
                total_size = 0
                for file_info in info['files']:
                    response += f"• {file_info['name']} - {file_info['size_mb']} MB\n"
                    total_size += file_info['size_mb']
                response += f"\n**Total Size**: {total_size:.2f} MB"
            else:
                response += "**Status**: No storage files found\n"
                response += "**Action**: Run `python3 lightrag_indexer.py` to create the knowledge base"
        
        return [types.TextContent(type="text", text=response)]
        
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"Storage status error: {str(e)}"
        )]

# AEGIS tool handlers
async def handle_write_text(args: dict) -> List[types.TextContent]:
    """Handle write_text tool - generate or revise text with LLM+RAG."""
    mode = args.get("mode", "draft")
    doc_type = args.get("doc_type", "proposal")
    section_id = args.get("section_id", "")
    prompt = args.get("prompt", "")
    must_cover = args.get("must_cover", [])
    objectives = args.get("objectives", [])
    kpis = args.get("kpis", [])
    word_budget = args.get("word_budget", 800)
    tone = args.get("tone", {})
    revise = args.get("revise", {})
    retrieval = args.get("retrieval", {})
    
    # Build retrieval query from context
    query_parts = [prompt, section_id] + must_cover + objectives
    query = " ".join(filter(None, query_parts))
    
    # Retrieve context
    top_k = retrieval.get("top_k", 15)
    raw_context = await query_engine.query_context_only(query, "hybrid")
    passages = query_engine.format_retrieval_passages(raw_context, top_k)
    
    # Build user prompt
    user_prompt = f"""Task: {mode}
Doc Type: {doc_type}
Section ID: {section_id}
Word Budget: {word_budget}
Tone Profile: {json.dumps(tone)}
Must Cover: {json.dumps(must_cover)}
Objectives: {json.dumps(objectives)}
KPIs: {json.dumps(kpis)}

Prompt: {prompt}

"""
    
    if mode == 'revise':
        user_prompt += f"""Previous Text:
<<<
{revise.get('previous_text', '')}
>>>
Revision Instructions: {json.dumps(revise.get('instructions', []))}

"""
    
    user_prompt += """# Retrieved Evidence (cite by id)
"""
    for passage in passages:
        user_prompt += f"""- [{passage['id']}] ({passage['doc_class']}) {passage['source_uri']}
  "{passage['snippet']}"
"""
    
    user_prompt += """
# Output JSON shape
{
  "text": "string (final draft with [R*] anchors)",
  "citations": [{"id":"R*","source_uri":"string","snippet":"string"}],
  "coverage": {"<must_cover item>": 0.0-1.0, ...},
  "retrieval_used": [{"id":"R*","doc_class":"string","source_uri":"string"}],
  "open_questions": ["string", ...]
}"""
    
    # Get LLM response
    response = await query_engine.llm_call(WRITE_TEXT_SYSTEM, user_prompt)
    
    return [types.TextContent(type="text", text=response)]

async def handle_assess_tone(args: dict) -> List[types.TextContent]:
    """Handle assess_tone tool - LLM-only style assessment."""
    text = args.get("text", "")
    profile = args.get("profile", "eu_proposal")
    targets = args.get("targets", {})
    custom_rubric = args.get("custom_rubric", [])
    
    # Build user prompt
    user_prompt = f"""Profile: {profile}
Targets: {json.dumps(targets)}
Custom Rubric: {json.dumps(custom_rubric)}

Text to Assess:
<<<
{text}
>>>

# Output JSON shape
{{
  "scores": {{"clarity":0-1,"formality":0-1,"vagueness":0-1,"conciseness":0-1}},
  "comments": ["string", ...],
  "edit_instructions": [
    {{"reason":"string","span":"string","suggest":"string"}}
  ],
  "meets_profile": true
}}"""
    
    # Get LLM response
    response = await query_engine.llm_call(ASSESS_TONE_SYSTEM, user_prompt)
    
    return [types.TextContent(type="text", text=response)]

async def handle_tech_assess(args: dict) -> List[types.TextContent]:
    """Handle tech_assess tool - LLM+RAG technical validation."""
    text = args.get("text", "")
    project_id = args.get("project_id", "")
    trl_target = args.get("trl_target", 6)
    retrieval = args.get("retrieval", {})
    
    # Build retrieval query from technical claims in text
    query = f"AEGIS tools assets technical capabilities {text[:200]}"
    
    # Retrieve technical context
    top_k = retrieval.get("top_k", 20)
    raw_context = await query_engine.query_context_only(query, "hybrid")
    passages = query_engine.format_retrieval_passages(raw_context, top_k)
    
    # Build user prompt
    user_prompt = f"""Project ID: {project_id}
Target TRL: {trl_target}

Text to Check:
<<<
{text}
>>>

# Retrieved AEGIS Evidence
"""
    for passage in passages:
        user_prompt += f"""- [{passage['id']}] ({passage['doc_class']}) {passage['source_uri']}
  "{passage['snippet']}"
"""
    
    user_prompt += f"""
# Output JSON shape
{{
  "findings": [{{"claim":"string","evidence":"R* | none","confidence":0-1}}],
  "asset_matches": [{{"asset":"string","fit":"strong|medium|weak","evidence":"R*|none","next_step":"string"}}],
  "trl": {{"current": 1-9, "target": {trl_target}, "rationale":"string"}},
  "gaps": ["string", ...],
  "overpromises": ["string", ...],
  "required_evidence": ["string", ...],
  "citations": [{{"id":"R*","source_uri":"string","snippet":"string"}}]
}}"""
    
    # Get LLM response
    response = await query_engine.llm_call(TECH_ASSESS_SYSTEM, user_prompt)
    
    return [types.TextContent(type="text", text=response)]

async def main():
    """Run unified MCP server."""
    logger.info("=== LightRAG Unified MCP Server Starting ===")
    logger.info("Available tools: retrieval, storage_status, write_text, assess_tone, tech_assess")
    
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream,
            InitializationOptions(
                server_name="lightrag-unified-mcp",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
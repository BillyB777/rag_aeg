#!/usr/bin/env python3
"""
AEGIS Minimal MCP Server
========================

Minimal LLM-first MCP interface with exactly 3 tools:
- write_text: Generate/revise text with LLM+RAG
- assess_tone: LLM-only style assessment 
- tech_assess: LLM+RAG technical validation

All logic is LLM reasoning + LightRAG retrieval. No hard rules.
"""

import asyncio
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Any
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

# MCP imports
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aegis-minimal-mcp")

# LLM Prompt Templates
WRITE_TEXT_SYSTEM = """You are an AEGIS proposal writer. Draft or revise text for EU Horizon documents (proposals, deliverables, concept notes) using only the provided retrieval context. You must:
- Cover all requested bullets concisely within the word budget.
- Ground key claims in the retrieved evidence; add inline anchors like [R1], [R2].
- Prefer precise commitments and concrete KPIs over vague promises.
- If required evidence is missing, surface it in `open_questions` instead of guessing.
- Output STRICT JSON only."""

ASSESS_TONE_SYSTEM = """You are an AEGIS senior editor. Assess style and tone of EU Horizon text purely by expert judgement (no deterministic metrics). Score against the requested profile and provide actionable edit instructions. Output STRICT JSON only."""

TECH_ASSESS_SYSTEM = """You are an AEGIS technical reviewer. Validate claims against AEGIS assets and TRL progression using only retrieved context. Identify matches, gaps, overpromises, and required evidence to reach the target TRL. Prefer conservative, evidence-backed judgements. Output STRICT JSON only."""

class AEGISMinimalEngine:
    def __init__(self, working_dir: str = None):
        self.working_dir = working_dir or os.getenv("LIGHTRAG_STORAGE_DIR", "./lightrag_storage")
        self.rag = None
        self.initialized = False
        
        # Check if LightRAG storage exists
        if not os.path.exists(self.working_dir):
            logger.error(f"LightRAG storage not found: {self.working_dir}")
    
    async def initialize(self):
        """Initialize LightRAG instance."""
        if self.initialized:
            return
        
        try:
            self.rag = LightRAG(
                working_dir=self.working_dir,
                embedding_func=openai_embed,
                llm_model_func=gpt_4o_mini_complete,
            )
            
            await self.rag.initialize_storages()
            await initialize_pipeline_status()
            
            self.initialized = True
            logger.info("AEGIS Minimal Engine initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise
    
    async def retrieve_context(self, query: str, filters: Dict = None, top_k: int = 15, chunk_top_k: int = 6) -> str:
        """Retrieve context from LightRAG and format for LLM prompts."""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Use hybrid mode with context-only retrieval
            result = await self.rag.aquery(
                query,
                param=QueryParam(
                    mode="hybrid",
                    only_need_context=True,
                    top_k=top_k,
                    chunk_top_k=chunk_top_k,
                    max_total_tokens=12000
                )
            )
            
            if not result or not result.strip():
                return "No relevant context found."
            
            return result
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return f"Retrieval error: {str(e)}"
    
    def format_retrieval_passages(self, raw_result: str, max_passages: int = 10) -> List[Dict[str, str]]:
        """Parse raw LightRAG result into structured passages."""
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
        """Make direct LLM call for tool logic."""
        try:
            # Format prompt for gpt_4o_mini_complete function
            full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"
            
            response = await gpt_4o_mini_complete(full_prompt, **{})
            return response
            
        except Exception as e:
            logger.error(f"LLM call error: {e}")
            return json.dumps({"error": f"LLM call failed: {str(e)}"})

# Global engine instance
engine = AEGISMinimalEngine()

# MCP Server
server = Server("aegis-minimal-mcp")

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """List the exactly 3 AEGIS tools."""
    return [
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

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> List[types.TextContent]:
    """Handle tool calls for the 3 AEGIS tools."""
    try:
        await engine.initialize()
        
        if name == "write_text":
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
    chunk_top_k = retrieval.get("chunk_top_k", 6) 
    raw_context = await engine.retrieve_context(query, retrieval.get("filters"), top_k, chunk_top_k)
    passages = engine.format_retrieval_passages(raw_context, top_k)
    
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
    response = await engine.llm_call(WRITE_TEXT_SYSTEM, user_prompt)
    
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
    response = await engine.llm_call(ASSESS_TONE_SYSTEM, user_prompt)
    
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
    raw_context = await engine.retrieve_context(query, retrieval.get("filters"), top_k, 6)
    passages = engine.format_retrieval_passages(raw_context, top_k)
    
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
    response = await engine.llm_call(TECH_ASSESS_SYSTEM, user_prompt)
    
    return [types.TextContent(type="text", text=response)]

async def main():
    """Run AEGIS Minimal MCP server."""
    logger.info("=== AEGIS Minimal MCP Server Starting ===")
    
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream,
            InitializationOptions(
                server_name="aegis-minimal-mcp",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
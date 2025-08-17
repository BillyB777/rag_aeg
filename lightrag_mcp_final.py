#!/usr/bin/env python3
"""
LightRAG MCP Server (Final Version)
===================================

MCP server for querying pre-built LightRAG storage.
Uses the same functions as indexing to ensure compatibility.
"""

import asyncio
import json
import logging
import sys
import os
from typing import List, Optional
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

# Set OpenAI API key for LightRAG functions
os.environ["OPENAI_API_KEY"] = "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"

# Import LightRAG components with required functions
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("lightrag-mcp")

class LightRAGQueryEngine:
    def __init__(self, working_dir: str = "./lightrag_storage"):
        self.working_dir = working_dir
        self.rag = None
        self.initialized = False
        
        # Check if LightRAG storage exists
        if not os.path.exists(working_dir):
            logger.error(f"LightRAG storage not found: {working_dir}")
    
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
            logger.info("LightRAG initialized for querying")
            
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

# Global instance
query_engine = LightRAGQueryEngine()

# MCP Server
server = Server("lightrag-mcp")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    try:
        return [
            types.Tool(
                name="retrieval",
                description="Search the CoEvolution project knowledge base using LightRAG. Returns relevant context from indexed documents.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Question or search terms about the CoEvolution project"
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
            types.Tool(
                name="storage_status",
                description="Check the status of the LightRAG storage and indexed documents",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            )
        ]
    except Exception:
        logger.exception("list_tools failed")
        return []  # don't crash the handshake
@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    try:
        if name == "retrieval":
            query = arguments.get("query", "")
            mode = arguments.get("mode", "hybrid")
            use_generation = arguments.get("use_generation", False)
            
            if not query:
                return [types.TextContent(type="text", text="Query is required")]
            
            try:
                if use_generation:
                    # Use LLM generation (requires OpenAI API)
                    result = await query_engine.query_with_generation(query, mode)
                    response_type = "Generated Response"
                else:
                    # Context-only (no additional API calls)
                    result = await query_engine.query_context_only(query, mode)
                    response_type = "Context Only"
                
                response = f"**CoEvolution LightRAG Search**\n\n"
                response += f"**Query**: {query}\n"
                response += f"**Mode**: {mode}\n"
                response += f"**Type**: {response_type}\n\n"
                response += f"**Result**:\n{result}"
                
                return [types.TextContent(type="text", text=response)]
                
            except Exception as e:
                error_msg = f"LightRAG search error: {str(e)}\n\n"
                if "storage" in str(e).lower() or "not found" in str(e).lower():
                    error_msg += "Please ensure the LightRAG indexing was completed successfully by running:\n"
                    error_msg += "python3 lightrag_indexer.py"
                
                return [types.TextContent(type="text", text=error_msg)]
        elif name == "storage_status":
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
        
        raise ValueError(f"Unknown tool: {name}")
    except Exception:
        logger.exception("call_tool failed")
        return [types.TextContent(type="text", text="Internal error. See server logs.")] 

async def main():
    """Run MCP server."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream,
            InitializationOptions(
                server_name="lightrag-mcp",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
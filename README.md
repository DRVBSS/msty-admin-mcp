# Msty Admin MCP

**AI-Powered Administration for Msty Studio Desktop 2.4.0+**

An MCP (Model Context Protocol) server that transforms Claude into an intelligent system administrator for [Msty Studio Desktop](https://msty.ai). Query databases, manage configurations, orchestrate local AI models, and build tiered AI workflows—all through natural conversation.

[![Version](https://img.shields.io/badge/version-7.0.0-blue.svg)](https://github.com/DRVBSS/msty-admin-mcp/releases)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-yellow.svg)](https://python.org)
[![Platform](https://img.shields.io/badge/platform-macOS-lightgrey.svg)](https://apple.com)
[![Msty](https://img.shields.io/badge/Msty-2.4.0+-purple.svg)](https://msty.ai)
[![Tests](https://img.shields.io/badge/tests-92%20passing-brightgreen.svg)](tests/)
[![Tools](https://img.shields.io/badge/tools-77-orange.svg)](src/)

> **v7.0.0** - Massive expansion with 77 tools including Knowledge Stacks, Model Management, Claude↔Local Bridge, Turnstile Workflows, Live Context, and Conversation Analytics.

---

## What's New in v7.0.0

### 35 New Tools Across 6 New Phases

| Phase | Tools | Description |
|-------|-------|-------------|
| **Phase 10: Knowledge Stacks** | 5 | RAG system management - list, search, analyze |
| **Phase 11: Model Management** | 6 | Download/delete models, find duplicates, storage analysis |
| **Phase 12: Claude↔Local Bridge** | 5 | Intelligent model delegation, multi-model consensus |
| **Phase 13: Turnstile Workflows** | 7 | 5 built-in automation templates, dry-run execution |
| **Phase 14: Live Context** | 5 | Real-time system/datetime/Msty context for prompts |
| **Phase 15: Conversation Analytics** | 5 | Usage patterns, content analysis, session metrics |

### New Modules

| Module | Purpose |
|--------|---------|
| `knowledge_stacks.py` | Knowledge Stack management for RAG |
| `model_management.py` | Model download/delete operations |
| `model_bridge.py` | Claude↔Local Model orchestration |
| `turnstiles.py` | Turnstile workflow automation |
| `live_context.py` | Real-time context providers |
| `conversation_analytics.py` | Advanced analytics |
| `server_extensions.py` | Extension tool registration |

### Enhanced Tagging System v2.0
- Context length awareness: `long_context` (100K+), `very_long_context` (250K+), `massive_context` (500K+)
- Quantization detection: `fp16`, `8bit`, `6bit`, `5bit`, `4bit`, `3bit`
- Architecture tags: `moe`, `mlx`, `gguf`
- New size tier: `massive` (200B+ parameters)

### Comprehensive Testing
- **92 unit tests** covering all modules
- SQL injection protection verified
- Error handling tested
- Cache behavior validated

### Msty 2.4.0+ Service Support
| Service | Port | Description |
|---------|------|-------------|
| **Local AI Service** | 11964 | Ollama-compatible API |
| **MLX Service** | 11973 | Apple Silicon optimized models |
| **LLaMA.cpp Service** | 11454 | GGUF model support |
| **Vibe CLI Proxy** | 8317 | Unified proxy for all AI services |

---

## What is This?

Msty Admin MCP lets you manage your entire Msty Studio installation through Claude Desktop. Instead of clicking through menus or manually editing config files, just ask Claude:

> "Show me my Msty personas and suggest improvements"

> "Compare my local models on a coding task"

> "What models do I have available across all services?"

> "Benchmark my fastest model for coding tasks"

Claude handles the rest—querying databases, calling APIs, analysing results, and presenting actionable insights.

---

## Quick Start

### Prerequisites

- **macOS** (Apple Silicon recommended)
- **Python 3.10+**
- **[Msty Studio Desktop 2.4.0+](https://msty.ai)** installed
- **Claude Desktop** with MCP support

### Installation

```bash
# Clone the repository
git clone https://github.com/DRVBSS/msty-admin-mcp.git
cd msty-admin-mcp

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Claude Desktop Configuration

**Important**: Claude Desktop doesn't always respect the `cwd` setting, so we use a shell script launcher.

1. The repository includes `run_msty_server.sh`. Make sure it's executable:
   ```bash
   chmod +x run_msty_server.sh
   ```

2. Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:
   ```json
   {
     "mcpServers": {
       "msty-admin": {
         "command": "/absolute/path/to/msty-admin-mcp/run_msty_server.sh",
         "env": {
           "MSTY_TIMEOUT": "30"
         }
       }
     }
   }
   ```

3. **Restart Claude Desktop** (Cmd+Q, then reopen)

4. You should see "msty-admin" in your available tools with **42 tools** loaded.

---

## Available Tools (42 Total)

### Phase 1: Installation & Health (7 tools)
| Tool | Description |
|------|-------------|
| `detect_msty_installation` | Find Msty Studio, verify paths, check running status |
| `read_msty_database` | Query conversations, personas, prompts, tools |
| `list_configured_tools` | View MCP toolbox configuration |
| `get_model_providers` | List AI providers and local models |
| `analyse_msty_health` | Database integrity, storage, all 4 service status |
| `get_server_status` | MCP server info and capabilities |
| `scan_database_locations` | Find database files in common locations |

### Phase 2: Configuration Management (4 tools)
| Tool | Description |
|------|-------------|
| `export_tool_config` | Export MCP configs for backup or sync |
| `import_tool_config` | Validate and prepare tools for Msty import |
| `generate_persona` | Create personas from templates (opus, coder, writer, minimal) |
| `sync_claude_preferences` | Convert Claude Desktop preferences to Msty persona |

### Phase 3: Local Model Integration (8 tools)
| Tool | Description |
|------|-------------|
| `get_sidecar_status` | Check all 4 services (Local AI, MLX, LLaMA.cpp, Vibe Proxy) |
| `list_available_models` | Query models from ALL services with breakdown |
| `query_local_ai_service` | Direct low-level API access |
| `chat_with_local_model` | Send messages with automatic metric tracking |
| `recommend_model` | Hardware-aware model recommendations by use case |
| `list_model_tags` | Get available tags for smart model selection |
| `find_model_by_tag` | Find models matching specific tags |
| `get_cache_stats` | View response cache statistics |
| `clear_cache` | Clear cached responses |

### Phase 4: Intelligence & Analytics (5 tools)
| Tool | Description |
|------|-------------|
| `get_model_performance_metrics` | Tokens/sec, latency, error rates over time |
| `analyse_conversation_patterns` | Privacy-respecting usage analytics |
| `compare_model_responses` | Same prompt to multiple models, compare quality/speed |
| `optimise_knowledge_stacks` | Analyse and recommend improvements |
| `suggest_persona_improvements` | AI-powered persona optimisation |

### Phase 5: Calibration & Workflow (4 tools)
| Tool | Description |
|------|-------------|
| `run_calibration_test` | Test models across categories with quality scoring |
| `evaluate_response_quality` | Score any response using heuristic evaluation |
| `identify_handoff_triggers` | Track patterns that should escalate to Claude |
| `get_calibration_history` | Historical results with trends and statistics |

### Phase 6: Advanced Model Management (4 tools)
| Tool | Description |
|------|-------------|
| `get_model_details` | Comprehensive model info (context length, parameters, tags, capabilities) |
| `benchmark_model` | Performance benchmarks at different context sizes (tokens/sec) |
| `list_local_model_files` | List MLX and GGUF model files on disk with sizes |
| `estimate_model_requirements` | Estimate memory/hardware requirements for a model |

### Phase 7: Conversation Management (3 tools)
| Tool | Description |
|------|-------------|
| `export_conversations` | Export chat history in JSON, Markdown, or CSV format |
| `search_conversations` | Search through conversations by keyword or title |
| `get_conversation_stats` | Usage analytics: messages per day, model usage, session lengths |

### Phase 8: Prompt Templates & Automation (4 tools)
| Tool | Description |
|------|-------------|
| `create_prompt_template` | Create reusable templates with `{{variable}}` placeholders |
| `list_prompt_templates` | List all templates, optionally filtered by category |
| `run_prompt_template` | Execute a template with variable substitutions |
| `smart_model_router` | Auto-select the best model for a given task description |

### Phase 9: Backup & System Management (3 tools)
| Tool | Description |
|------|-------------|
| `backup_configuration` | Create comprehensive backup of personas, prompts, templates, tools |
| `restore_configuration` | Restore configuration from a backup file |
| `get_system_resources` | CPU, memory, and disk usage relevant to AI inference |

---

## Model Tagging System

Msty Admin MCP includes a smart model tagging system with 60+ model-specific overrides for accurate routing.

### Available Tags

| Category | Tags | Description |
|----------|------|-------------|
| **Size** | `large`, `medium`, `small` | Model parameter count (70B+, 13-34B, <13B) |
| **Speed** | `fast` | Quick response models (Haiku, Flash, Mini, etc.) |
| **Capability** | `coding`, `reasoning`, `creative`, `vision`, `embedding` | Specialized capabilities |
| **Context** | `long_context` | Models with 128K+ context windows |
| **Quality** | `quality` | High-quality output models |
| **General** | `general` | General-purpose models |

### Finding Models by Tag

```
You: Find me a fast coding model

Claude: Using find_model_by_tag with tag="coding" and prefer_fast=true...

        Found 3 fast coding models:
        1. deepseek-coder-v2-lite (fast, coding, small)
        2. codestral-mini (fast, coding, small)
        3. starcoder2-3b (fast, coding, small)
```

### Smart Model Router

```
You: What's the best model for writing a poem?

Claude: Using smart_model_router with task="writing a poem"...

        Recommended: claude-3-opus (tags: quality, creative, reasoning, large)
        Reasoning: Creative writing tasks benefit from models with
        the 'creative' tag and high quality output.
```

---

## Usage Examples

### Check Service Status

```
You: What's the status of my Msty services?

Claude: All 4 services are running:
        ✅ Local AI Service (port 11964) - 2 models
        ✅ MLX Service (port 11973) - 11 models
        ✅ LLaMA.cpp Service (port 11454) - 10 models
        ✅ Vibe CLI Proxy (port 8317) - 36 models

        Total: 59 models available
```

### Benchmark a Model

```
You: Benchmark my fastest coding model

Claude: Running benchmark on deepseek-coder-v2-lite...

        Results (3 runs each):
        | Context | Tokens/sec | Latency |
        |---------|------------|---------|
        | 50 tokens | 42.3 t/s | 1.2s |
        | 200 tokens | 38.7 t/s | 5.2s |
        | 500 tokens | 35.1 t/s | 14.3s |

        Average: 38.7 tokens/sec
```

### Export Conversations

```
You: Export my last week's conversations to markdown

Claude: Exported 23 conversations from the last 7 days.

        File: conversations_2026-01-26.md
        Format: Markdown with headers per conversation
        Total messages: 156
```

### Create Prompt Template

```
You: Create a code review template

Claude: Created template "code-review":

        Template: "Review this {{language}} code for {{focus}}:

                   ```{{language}}
                   {{code}}
                   ```

                   Provide specific suggestions."

        Variables: language, focus, code
        Category: coding
```

---

## Environment Variables

Customize behavior with these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MSTY_SIDECAR_HOST` | `127.0.0.1` | Service host address |
| `MSTY_AI_PORT` | `11964` | Local AI Service port |
| `MSTY_MLX_PORT` | `11973` | MLX Service port |
| `MSTY_LLAMACPP_PORT` | `11454` | LLaMA.cpp Service port |
| `MSTY_VIBE_PORT` | `8317` | Vibe CLI Proxy port |
| `MSTY_TIMEOUT` | `10` | API request timeout (seconds) |
| `MSTY_DATABASE_PATH` | (auto-detect) | Override database path |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Claude Desktop                          │
│                           │                                  │
│                      MCP Protocol                            │
│                           │                                  │
│                ┌──────────┴──────────┐                      │
│                ▼                     ▼                      │
│      ┌─────────────────┐   ┌─────────────────┐             │
│      │ Msty Admin MCP  │   │  Other MCPs     │             │
│      │   (42 tools)    │   │                 │             │
│      └────────┬────────┘   └─────────────────┘             │
└───────────────┼─────────────────────────────────────────────┘
                │
     ┌──────────┴──────────────────────────────┐
     ▼                                         ▼
┌──────────┐                          ┌──────────────────┐
│  Msty    │                          │   Msty Studio    │
│ Database │                          │   2.4.0+ App     │
│ (SQLite) │                          └────────┬─────────┘
└──────────┘                                   │
                          ┌────────────────────┼────────────────────┐
                          ▼                    ▼                    ▼
                   ┌────────────┐      ┌────────────┐      ┌────────────┐
                   │ Local AI   │      │    MLX     │      │ LLaMA.cpp  │
                   │  :11964    │      │   :11973   │      │   :11454   │
                   └────────────┘      └────────────┘      └────────────┘
                          │                    │                    │
                          └────────────────────┼────────────────────┘
                                               ▼
                                        ┌────────────┐
                                        │ Vibe Proxy │
                                        │   :8317    │
                                        └────────────┘
```

---

## Project Structure

```
msty-admin-mcp/
├── src/
│   ├── __init__.py         # Package exports
│   ├── constants.py        # Configuration constants
│   ├── models.py           # Data classes
│   ├── errors.py           # Standardized error handling
│   ├── paths.py            # Path resolution utilities
│   ├── database.py         # SQL operations (injection protected)
│   ├── network.py          # API request helpers
│   ├── cache.py            # Response caching
│   ├── tagging.py          # Model tagging system
│   ├── server.py           # Main MCP server (42 tools)
│   └── phase4_5_tools.py   # Metrics and calibration
├── tests/
│   ├── test_server.py      # Integration tests
│   ├── test_constants.py   # Constants tests
│   ├── test_paths.py       # Path utilities tests
│   ├── test_database.py    # Database tests (SQL injection)
│   ├── test_network.py     # Network tests
│   ├── test_cache.py       # Cache tests
│   └── test_tagging.py     # Tagging tests
├── docs/
│   ├── API.md              # API reference & error codes
│   └── DEVELOPMENT.md      # Development guide
├── run_msty_server.sh      # Shell script launcher (required!)
├── requirements.txt
├── pyproject.toml
├── CHANGELOG.md
├── LICENSE
└── README.md
```

---

## Documentation

- **[API Reference](docs/API.md)** - Error codes, response formats, tool parameters
- **[Development Guide](docs/DEVELOPMENT.md)** - Contributing, testing, architecture
- **[Changelog](CHANGELOG.md)** - Version history and migration notes

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'src'"

Claude Desktop isn't running from the correct directory. Make sure you're using the shell script launcher (`run_msty_server.sh`) instead of calling Python directly.

### "No Local AI services are running"

1. Open Msty Studio
2. Go to Settings → Local AI / MLX / LLaMA.cpp
3. Make sure the services show "Running"

### Claude doesn't see the msty-admin tools

1. Check your `claude_desktop_config.json` has the correct absolute path
2. Make sure `run_msty_server.sh` is executable (`chmod +x`)
3. Restart Claude Desktop completely (Cmd+Q, then reopen)

### Only seeing 2 embedding models

The models shown depend on which service responds first. Use `list_available_models` to see ALL models from all services with the `by_service` breakdown.

### Database not found

1. Run `detect_msty_installation` first to verify paths
2. Use `scan_database_locations` to find database files
3. Set `MSTY_DATABASE_PATH` environment variable if needed

---

## Security

See [docs/API.md](docs/API.md#security) for details on:

- **SQL Injection Protection** - Table allowlists, parameterized queries
- **API Key Handling** - Keys are never logged or returned
- **Network Security** - All calls to localhost, configurable timeouts

---

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest tests/ -v`)
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for detailed contribution guidelines.

---

## Credits

- **Original Author**: [Pineapple](https://github.com/M-Pineapple) - Created the original msty-admin-mcp
- **v5.0.0+ Fork**: [DBSS](https://github.com/DRVBSS) - Msty 2.4.0+ compatibility, modular architecture

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [Msty Studio](https://msty.ai) - The excellent local AI application this MCP administers
- [Anthropic](https://anthropic.com) - For Claude and the MCP protocol
- [Model Context Protocol](https://modelcontextprotocol.io) - The foundation making this possible

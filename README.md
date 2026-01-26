# Msty Admin MCP

**AI-Powered Administration for Msty Studio Desktop 2.4.0+**

An MCP (Model Context Protocol) server that transforms Claude into an intelligent system administrator for [Msty Studio Desktop](https://msty.ai). Query databases, manage configurations, orchestrate local AI models, and build tiered AI workflows—all through natural conversation.

[![Version](https://img.shields.io/badge/version-5.0.0-blue.svg)](https://github.com/DRVBSS/msty-admin-mcp/releases)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-yellow.svg)](https://python.org)
[![Platform](https://img.shields.io/badge/platform-macOS-lightgrey.svg)](https://apple.com)
[![Msty](https://img.shields.io/badge/Msty-2.4.0+-purple.svg)](https://msty.ai)

> **v5.0.0** - Full support for Msty 2.4.0+ architecture with Local AI, MLX, LLaMA.cpp, and Vibe Proxy services.

---

## What's New in v5.0.0

This version completely rewrites service detection for **Msty 2.4.0+**, which has Local AI services built directly into the main app (no separate Sidecar needed):

| Service | Port | Description |
|---------|------|-------------|
| **Local AI Service** | 11964 | Ollama-compatible API |
| **MLX Service** | 11973 | Apple Silicon optimized models |
| **LLaMA.cpp Service** | 11454 | GGUF model support |
| **Vibe CLI Proxy** | 8317 | Unified proxy for all AI services |

**Key Changes:**
- Detects all 4 services automatically
- Lists models from ALL services (not just one)
- No longer requires `MstySidecar` process
- Includes shell script launcher for reliable Claude Desktop integration

---

## What is This?

Msty Admin MCP lets you manage your entire Msty Studio installation through Claude Desktop. Instead of clicking through menus or manually editing config files, just ask Claude:

> "Show me my Msty personas and suggest improvements"

> "Compare my local models on a coding task"

> "What models do I have available across all services?"

> "What's the health status of my Msty installation?"

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

4. You should see "msty-admin" in your available tools with 24 tools loaded.

---

## Available Tools (24 Total)

### Phase 1: Installation & Health
| Tool | Description |
|------|-------------|
| `detect_msty_installation` | Find Msty Studio, verify paths, check running status |
| `read_msty_database` | Query conversations, personas, prompts, tools |
| `list_configured_tools` | View MCP toolbox configuration |
| `get_model_providers` | List AI providers and local models |
| `analyse_msty_health` | Database integrity, storage, all 4 service status |
| `get_server_status` | MCP server info and capabilities |

### Phase 2: Configuration Management
| Tool | Description |
|------|-------------|
| `export_tool_config` | Export MCP configs for backup or sync |
| `import_tool_config` | Validate and prepare tools for Msty import |
| `generate_persona` | Create personas from templates (opus, coder, writer, minimal) |
| `sync_claude_preferences` | Convert Claude Desktop preferences to Msty persona |

### Phase 3: Local Model Integration
| Tool | Description |
|------|-------------|
| `get_sidecar_status` | Check all 4 services (Local AI, MLX, LLaMA.cpp, Vibe Proxy) |
| `list_available_models` | Query models from ALL services with breakdown |
| `query_local_ai_service` | Direct low-level API access |
| `chat_with_local_model` | Send messages with automatic metric tracking |
| `recommend_model` | Hardware-aware model recommendations by use case |

### Phase 4: Intelligence & Analytics
| Tool | Description |
|------|-------------|
| `get_model_performance_metrics` | Tokens/sec, latency, error rates over time |
| `analyse_conversation_patterns` | Privacy-respecting usage analytics |
| `compare_model_responses` | Same prompt to multiple models, compare quality/speed |
| `optimise_knowledge_stacks` | Analyse and recommend improvements |
| `suggest_persona_improvements` | AI-powered persona optimisation |

### Phase 5: Calibration & Workflow
| Tool | Description |
|------|-------------|
| `run_calibration_test` | Test models across categories with quality scoring |
| `evaluate_response_quality` | Score any response using heuristic evaluation |
| `identify_handoff_triggers` | Track patterns that should escalate to Claude |
| `get_calibration_history` | Historical results with trends and statistics |

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

### List All Models

```
You: What models do I have?

Claude: Found 59 models across all services:

        MLX (Apple Silicon):
        - Qwen3-235B-A22B-8bit
        - Hermes-4-405B-MLX-6bit
        - DeepSeek-Coder-V2-Instruct-Q4
        ...

        LLaMA.cpp (GGUF):
        - DeepSeek-V3-0324-UD-Q4_K_XL
        - Llama-3.3-70B-Instruct-Q8_0
        - Nemotron-Ultra-253B
        ...
```

### Health Check

```
You: Check the health of my Msty installation

Claude: Health Status: ✅ Healthy

        Msty Studio: Running ✅
        Local AI Service (port 11964): Running ✅
        MLX Service (port 11973): Running ✅
        LLaMA.cpp Service (port 11454): Running ✅

        No issues detected.
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
│      │   (24 tools)    │   │                 │             │
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

---

## Project Structure

```
msty-admin-mcp/
├── src/
│   ├── __init__.py
│   ├── server.py           # Main MCP server (24 tools)
│   └── phase4_5_tools.py   # Metrics and calibration utilities
├── tests/
│   └── test_server.py
├── run_msty_server.sh      # Shell script launcher (required!)
├── requirements.txt
├── pyproject.toml
├── CHANGELOG.md
├── LICENSE
└── README.md
```

---

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Credits

- **Original Author**: [Pineapple](https://github.com/M-Pineapple) - Created the original msty-admin-mcp
- **v5.0.0 Fork**: [DBSS](https://github.com/DRVBSS) - Msty 2.4.0+ compatibility updates

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [Msty Studio](https://msty.ai) - The excellent local AI application this MCP administers
- [Anthropic](https://anthropic.com) - For Claude and the MCP protocol
- [Model Context Protocol](https://modelcontextprotocol.io) - The foundation making this possible

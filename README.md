# 🍍 Msty Admin MCP

**AI-Powered Administration for Msty Studio Desktop**

An MCP (Model Context Protocol) server that transforms Claude into an intelligent system administrator for [Msty Studio Desktop](https://msty.ai). Query databases, manage configurations, orchestrate local AI models, and build tiered AI workflows—all through natural conversation.

[![Version](https://img.shields.io/badge/version-4.0.0-blue.svg)](https://github.com/M-Pineapple/msty-admin-mcp/releases)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-yellow.svg)](https://python.org)
[![Platform](https://img.shields.io/badge/platform-macOS-lightgrey.svg)](https://apple.com)

---

## What is This?

Msty Admin MCP lets you manage your entire Msty Studio installation through Claude Desktop. Instead of clicking through menus or manually editing config files, just ask Claude:

> "Show me my Msty personas and suggest improvements"

> "Compare my local models on a coding task"

> "Run calibration tests to see which model handles reasoning best"

> "What's the health status of my Msty installation?"

Claude handles the rest—querying databases, calling APIs, analysing results, and presenting actionable insights.

---

## Use Cases

### 🔍 **1. Database Inspection & Insights**
Query your Msty database directly through conversation. Access conversations, personas, prompts, knowledge stacks, and MCP tools without touching SQLite.

```
"Show me all my Msty personas"
"How many conversations do I have?"
"List my configured MCP tools"
```

### 🏥 **2. Health Monitoring & Diagnostics**
Comprehensive health checks for your Msty installation—database integrity, storage usage, model cache status, and actionable recommendations.

```
"Check the health of my Msty installation"
"Is Sidecar running?"
"How much storage are my models using?"
```

### ⚙️ **3. Configuration Sync Between Claude & Msty**
Export MCP tool configurations from Claude Desktop and prepare them for Msty import. Generate personas from templates. Convert your Claude preferences to Msty format.

```
"Export my Claude Desktop MCP tools"
"Generate an Opus-style persona for Msty"
"Sync my preferences to Msty format"
```

### 🤖 **4. Local Model Orchestration**
Direct integration with Msty's Sidecar API. Chat with local models, compare responses across models, and get hardware-aware recommendations.

```
"List my available local models"
"Chat with qwen2.5:7b about Python async"
"Which model is best for coding on my hardware?"
```

### 📊 **5. Performance Analytics**
Track tokens per second, latency, and error rates across your local models. Privacy-respecting conversation analytics. Identify usage patterns.

```
"How fast are my local models?"
"Show performance metrics for the last 30 days"
"Which model has the best success rate?"
```

### 🎯 **6. Model Calibration & Quality Testing**
Test your local models against standardised prompts across categories (reasoning, coding, writing, analysis, creative). Score response quality. Track improvement over time.

```
"Run calibration tests on my Qwen model"
"Test my models on reasoning tasks"
"Show my calibration history"
```

### 🔄 **7. Tiered AI Workflow (Claude + Local)**
Identify which tasks your local models handle well and which should escalate to Claude. Build efficient hybrid workflows where simple tasks go local and complex tasks go to Claude.

```
"What tasks should I hand off to Claude?"
"Identify patterns where local models fail"
"Compare Claude vs local on this task"
```

### 🔬 **8. Database Discovery (Advanced)**
Through this MCP, we discovered Msty's internal database structure at:
```
~/Library/Application Support/MstyStudio/File System/000/t/00/00000000
```

This SQLite database contains tables for:
- `personas` - Your configured personas
- `tools` - MCP tool configurations
- `toolConfigs` - Tool parameters
- `conversationTexts` - Chat history
- `knowledgeStacks` - RAG configurations
- And more...

**Note**: Direct database manipulation is possible when Msty is closed, but unsupported. Use at your own risk.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **24 Tools** | Comprehensive administration toolkit |
| **Read-Only by Default** | Never writes to Msty's database |
| **Performance Tracking** | Automatic metrics for all local model calls |
| **Calibration System** | Built-in quality testing framework |
| **Hardware-Aware** | Recommendations based on your Mac's specs |
| **Privacy-Respecting** | No data sent externally |

---

## Available Tools (24 Total)

### Phase 1: Installation & Health
| Tool | What It Does |
|------|--------------|
| `detect_msty_installation` | Find Msty Studio, verify paths, check running status |
| `read_msty_database` | Query conversations, personas, prompts, tools |
| `list_configured_tools` | View MCP toolbox configuration |
| `get_model_providers` | List AI providers and local models |
| `analyse_msty_health` | Database integrity, storage, model cache, recommendations |
| `get_server_status` | MCP server info and capabilities |

### Phase 2: Configuration Management
| Tool | What It Does |
|------|--------------|
| `export_tool_config` | Export MCP configs for backup or sync |
| `import_tool_config` | Validate and prepare tools for Msty import |
| `generate_persona` | Create personas from templates (opus, coder, writer, minimal) |
| `sync_claude_preferences` | Convert Claude Desktop preferences to Msty persona |

### Phase 3: Local Model Integration
| Tool | What It Does |
|------|--------------|
| `get_sidecar_status` | Check Sidecar and Local AI Service health |
| `list_available_models` | Query models via Ollama-compatible API |
| `query_local_ai_service` | Direct low-level API access |
| `chat_with_local_model` | Send messages with automatic metric tracking |
| `recommend_model` | Hardware-aware model recommendations by use case |

### Phase 4: Intelligence & Analytics
| Tool | What It Does |
|------|--------------|
| `get_model_performance_metrics` | Tokens/sec, latency, error rates over time |
| `analyse_conversation_patterns` | Privacy-respecting usage analytics |
| `compare_model_responses` | Same prompt to multiple models, compare quality/speed |
| `optimise_knowledge_stacks` | Analyse and recommend improvements |
| `suggest_persona_improvements` | AI-powered persona optimisation |

### Phase 5: Calibration & Workflow
| Tool | What It Does |
|------|--------------|
| `run_calibration_test` | Test models across categories with quality scoring |
| `evaluate_response_quality` | Score any response using heuristic evaluation |
| `identify_handoff_triggers` | Track patterns that should escalate to Claude |
| `get_calibration_history` | Historical results with trends and statistics |

---

## Installation

### Prerequisites

- **macOS** (Apple Silicon or Intel)
- **Python 3.10+**
- **[Msty Studio Desktop](https://msty.ai)** installed
- **Msty Sidecar** running (for local model features)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/M-Pineapple/msty-admin-mcp.git
cd msty-admin-mcp

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Claude Desktop Configuration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "msty-admin": {
      "command": "/absolute/path/to/msty-admin-mcp/.venv/bin/python",
      "args": ["-m", "src.server"],
      "cwd": "/absolute/path/to/msty-admin-mcp"
    }
  }
}
```

Restart Claude Desktop. You should see "msty-admin" in your available tools.

---

## Usage Examples

### Check Your Installation

```
You: What's the status of my Msty installation?

Claude: Your Msty Studio is installed at /Applications/MstyStudio.app 
        (version 2.2.0). Sidecar is running with 3 models available.
        Database is healthy at 45MB. No issues detected.
```

### Query Your Data

```
You: Show me my Msty personas

Claude: You have 3 personas configured:
        1. "Opus Assistant" - British English, quality focus (temp: 0.7)
        2. "Swift Dev Assistant" - Development focused (temp: 0.3)
        3. "Current Ventures" - Business operations (temp: 0.5)
```

### Compare Local Models

```
You: Compare my local models on this coding task: 
     "Write a Python function to find duplicates in a list"

Claude: Tested 2 models:
        
        qwen2.5-coder:7b
        ├─ Quality: 0.85 | Speed: 38 tok/s | Latency: 2.1s
        └─ Clean implementation with list comprehension
        
        qwen2.5:7b  
        ├─ Quality: 0.72 | Speed: 42 tok/s | Latency: 1.8s
        └─ Works but less elegant
        
        Winner (balanced): qwen2.5-coder:7b
```

### Run Calibration

```
You: Run calibration tests on my Qwen model for reasoning tasks

Claude: Calibration Results for qwen2.5:7b (Reasoning)
        
        Test 1: Bat and ball problem
        ├─ Score: 0.82 ✅ PASSED
        └─ Correctly identified $0.05
        
        Test 2: Widget machines problem  
        ├─ Score: 0.78 ✅ PASSED
        └─ Showed step-by-step reasoning
        
        Summary: 2/2 passed, average score 0.80
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Claude Desktop                        │
│                         │                                │
│                    MCP Protocol                          │
│                         │                                │
│              ┌──────────┴──────────┐                    │
│              ▼                     ▼                    │
│    ┌─────────────────┐   ┌─────────────────┐           │
│    │ Msty Admin MCP  │   │  Other MCPs     │           │
│    │   (24 tools)    │   │ (Memory, etc.)  │           │
│    └────────┬────────┘   └─────────────────┘           │
└─────────────┼───────────────────────────────────────────┘
              │
   ┌──────────┴──────────┐
   ▼                     ▼
┌──────────┐      ┌──────────────┐
│  Msty    │      │   Sidecar    │
│ Database │      │  Local AI    │
│ (SQLite) │      │   Service    │
└──────────┘      └──────────────┘
     │                   │
     │            ┌──────┴──────┐
     │            ▼             ▼
     │      ┌──────────┐  ┌──────────┐
     │      │ Qwen 2.5 │  │ Llama 3  │
     │      │   7B     │  │   8B     │
     │      └──────────┘  └──────────┘
     │
     ▼
┌────────────────────────────────────┐
│ ~/Library/Application Support/     │
│ MstyStudio/File System/000/t/00/   │
│ 00000000 (SQLite Database)         │
├────────────────────────────────────┤
│ Tables:                            │
│ • personas                         │
│ • tools                            │
│ • toolConfigs                      │
│ • conversationTexts                │
│ • knowledgeStacks                  │
│ • and more...                      │
└────────────────────────────────────┘
```

### Data Storage

| Location | Purpose |
|----------|---------|
| Msty Database | Read-only queries (conversations, personas, etc.) |
| `~/.msty-admin/` | MCP's own metrics and calibration data |

The MCP never writes to Msty's database—it only reads. All metrics and calibration results are stored separately.

---

## Hardware Recommendations

### For Basic Use (Inspection, Health Checks)
- Any Mac with Msty installed
- No local models required

### For Local Model Features
| RAM | Recommended Models | Quality |
|-----|-------------------|---------|
| 8GB | qwen2.5:3b, gemma3:4b | Basic |
| 16GB | qwen2.5:7b, qwen2.5-coder:7b | Good |
| 32GB | qwen2.5:14b, llama3.1:8b | Very Good |
| 64GB+ | qwen2.5:32b, mixtral:8x7b | Excellent |
| 128GB+ | qwen2.5:72b, llama3.1:70b | Near-Claude |

### Performance Expectations (Apple Silicon)

| Model | M1 Pro 16GB | M2 Max 64GB | M3 Max 128GB |
|-------|-------------|-------------|--------------|
| 7B | 30-45 tok/s | 50-70 tok/s | 60-80 tok/s |
| 14B | Slow | 30-45 tok/s | 45-60 tok/s |
| 32B | ❌ | 15-25 tok/s | 25-40 tok/s |
| 70B | ❌ | ❌ | 10-20 tok/s |

---

## FAQ

### General

**Q: Do I need Msty Studio Desktop installed?**  
A: Yes. This MCP is specifically designed to administer Msty Studio. Without it, most tools won't function.

**Q: Does this work on Windows or Linux?**  
A: Currently macOS only. Msty Studio Desktop is a macOS application.

**Q: Is my data safe?**  
A: The MCP only reads from Msty's database—it never writes to it. Metrics and calibration data are stored separately in `~/.msty-admin/`. No data is sent externally.

### Local Models

**Q: Do I need local models installed?**  
A: For basic features (database queries, health checks), no. For local model features (chat, compare, calibrate), you need Msty Sidecar running with at least one model.

**Q: Which local models work best?**  
A: Use `recommend_model` with your use case. Generally:
- **Coding**: qwen2.5-coder (7B or 32B depending on your RAM)
- **General**: qwen2.5 (7B for speed, 32B for quality)
- **Fast responses**: gemma3:4b or qwen3:0.6b

**Q: What's the Sidecar?**  
A: Msty Sidecar is the background service that hosts local models. It provides an Ollama-compatible API on port 11964.

**Q: Can local models use MCP tools?**  
A: Yes, but smaller models (7B and below) often struggle with complex tool orchestration. Models 14B+ handle tools much better. For reliable MCP tool usage, consider 32B+ models or stick with Claude for complex workflows.

### Calibration

**Q: What is calibration?**  
A: Calibration tests your local models against standardised prompts to measure quality. Categories include reasoning, coding, writing, analysis, and creative tasks.

**Q: What's a good calibration score?**  
A: Scores range 0.0-1.0. Generally:
- 0.8+ = Excellent
- 0.6-0.8 = Good (passes threshold)
- 0.4-0.6 = Fair
- Below 0.4 = Poor

**Q: What are handoff triggers?**  
A: Patterns that indicate a task should be handled by Claude instead of a local model. The MCP learns these from failed calibration tests.

### Troubleshooting

**Q: Claude doesn't see the msty-admin tools**  
A: Check your `claude_desktop_config.json` paths are absolute (not relative). Restart Claude Desktop after changes.

**Q: "Sidecar not running" error**  
A: Start Msty Sidecar from the Msty Studio menu bar icon, or ensure Msty is open.

**Q: "Database not found" error**  
A: Msty stores its database in `~/Library/Application Support/MstyStudio/File System/000/t/00/00000000`. Ensure Msty has been launched at least once.

**Q: Model comparison takes too long**  
A: Each model runs sequentially. Limit comparisons to 3-5 models. Larger models (32B+) take longer.

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
├── requirements.txt
├── pyproject.toml
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

## Roadmap

- [ ] Windows/Linux support (when Msty supports it)
- [ ] Direct persona import via API (pending Msty API)
- [ ] Automatic model download recommendations
- [ ] Integration with Ollama CLI
- [ ] Web UI for metrics dashboard

---

## License

MIT License - see [LICENSE](LICENSE) for details.


---

**Created by Pineapple 🍍**

*Making local AI administration effortless.*

# 🍍 Msty Admin MCP

**AI-Administered Msty Studio Desktop Management System**

An MCP (Model Context Protocol) server that enables Claude Desktop to act as an intelligent system administrator for Msty Studio Desktop, providing database insights, configuration management, hardware optimization, and seamless sync capabilities.

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/M-Pineapple/msty-admin-mcp)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-yellow.svg)](https://python.org)
[![Platform](https://img.shields.io/badge/platform-macOS-lightgrey.svg)](https://apple.com)

## 🎯 Overview

Msty Admin MCP bridges the gap between Claude Desktop and Msty Studio Desktop, enabling:

- **Database Insights**: Query conversations, personas, prompts, and tools directly
- **Health Monitoring**: Comprehensive system health analysis and recommendations
- **Configuration Sync**: Export/import tools between Claude Desktop and Msty
- **Hardware Optimization**: Apple Silicon-optimized model recommendations
- **Tiered AI Workflow**: Local model calibration with Claude Opus handoff

## 🚀 Features

### Phase 1: Foundational Tools ✅

| Tool | Description |
|------|-------------|
| `detect_msty_installation` | Find Msty Studio, verify version, locate data paths |
| `read_msty_database` | Query SQLite for conversations, personas, prompts, tools |
| `list_configured_tools` | Read MCP toolbox configuration |
| `get_model_providers` | List configured AI providers and local models |
| `analyse_msty_health` | Check database integrity, storage, model cache status |
| `get_server_status` | Server info and available capabilities |

### Phase 2: Configuration Management (Planned)

- `export_tool_config` - Generate MCP tool JSON for sync
- `generate_persona` - Create persona configurations
- `sync_claude_preferences` - Convert Claude Desktop prefs to Msty format
- `import_tool_config` - Import tool configurations into Msty

### Phase 3: Automation Bridge (Planned)

- `open_msty_studio` - Launch application programmatically
- `trigger_toolbox_import` - AppleScript automation for imports
- `backup_msty_data` - Create timestamped backups
- `restore_from_backup` - Restore previous configurations

### Phase 4: Intelligence Layer (Planned)

- `recommend_models_for_hardware` - Mac specs → optimal MLX models
- `analyse_conversation_patterns` - Usage insights
- `optimise_knowledge_stacks` - Performance recommendations
- `suggest_persona_improvements` - AI-powered persona optimization

### Phase 5: Local Model Calibration & Handoff (Planned)

The ultimate goal: run local MLX models that perform at Claude Opus level for routine tasks, with seamless escalation to Claude when complexity demands it.

```
┌─────────────────────────────────────────────────┐
│                 Task Incoming                    │
└─────────────────────┬───────────────────────────┘
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
┌─────────────────┐      ┌─────────────────┐
│   Local MLX     │      │   Claude Opus   │
│   (Routine)     │      │   (Complex)     │
│                 │      │                 │
│ • File ops      │      │ • Architecture  │
│ • Simple code   │      │ • Deep analysis │
│ • Status checks │      │ • Creative work │
│ • Data queries  │      │ • Multi-step    │
└────────┬────────┘      └────────┬────────┘
         │                        │
         └────────────┬───────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│              Shared MCP Ecosystem                │
│  Memory • Trello • GitHub • Filesystem • More   │
└─────────────────────────────────────────────────┘
```

**Calibration Protocol:**
1. Test prompts sent to both local model AND Claude Opus (benchmark)
2. Claude Opus evaluates local output against its own benchmark
3. Scoring: Accuracy, Reasoning, Style, Tool Use, Safety
4. Auto-tune recommendations applied to Msty persona
5. Iterate until local model achieves "Opus approval"

**Phase 5 Tools:**
- `create_opus_persona` - Generate Msty persona with Universal Preferences
- `sync_mcp_toolbox` - Mirror Claude Desktop MCP config to Msty
- `run_calibration_suite` - Execute test prompts against local model
- `evaluate_response_pair` - Opus compares local vs benchmark
- `generate_tuning_recommendations` - Auto-suggest config adjustments
- `apply_persona_adjustments` - Auto-update Msty persona
- `prepare_handoff_context` - Package state for Claude escalation
- `identify_handoff_triggers` - Learn which prompts need escalation
- `track_calibration_history` - Monitor improvement over iterations

## 📦 Installation

### Prerequisites

- macOS (Apple Silicon recommended)
- Python 3.10+
- Msty Studio Desktop installed
- Claude Desktop with MCP support

### Quick Install

```bash
# Clone the repository
git clone https://github.com/M-Pineapple/msty-admin-mcp.git
cd msty-admin-mcp

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### Claude Desktop Configuration

Add to your `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "msty-admin": {
      "command": "/path/to/msty-admin-mcp/.venv/bin/python",
      "args": ["-m", "src.server"],
      "cwd": "/path/to/msty-admin-mcp"
    }
  }
}
```

### Msty Studio Configuration

Import the tool via Toolbox → Add New Tool → Import from JSON:

```json
{
  "command": "python",
  "args": ["-m", "src.server"],
  "cwd": "/path/to/msty-admin-mcp",
  "env": {}
}
```

## 🔧 Usage

Once configured, ask Claude to use the Msty Admin tools:

### Check Installation

```
"Check if Msty Studio is installed and get its status"
```

### Database Insights

```
"Show me statistics about my Msty database"
"List all my configured personas in Msty"
"What MCP tools do I have in Msty's Toolbox?"
```

### Health Check

```
"Run a health check on my Msty installation"
"Is my Msty database healthy? Any recommendations?"
```

### Model Providers

```
"What AI models do I have configured in Msty?"
"List my local MLX models"
```

## 📁 Project Structure

```
msty-admin-mcp/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── server.py            # Main FastMCP server (Phase 1)
│   ├── database/            # Database operations (Phase 2+)
│   ├── config/              # Configuration sync (Phase 2+)
│   ├── hardware/            # Hardware detection (Phase 4)
│   ├── automation/          # AppleScript integration (Phase 3)
│   └── utils/               # Shared utilities
├── tests/
│   └── test_server.py       # Unit tests
├── pyproject.toml           # Modern Python packaging
├── requirements.txt         # Dependencies
├── LICENSE                  # MIT License
└── README.md               # This file
```

## 🗂️ Msty Studio Paths

The MCP server automatically detects these locations:

| Component | macOS Path |
|-----------|------------|
| Application | `/Applications/MstyStudio.app` |
| Data Directory | `~/Library/Application Support/MstyStudio/` |
| Database | `~/Library/Application Support/MstyStudio/msty.db` |
| MLX Models | `~/Library/Application Support/MstyStudio/models-mlx/` |
| Sidecar | `~/Library/Application Support/MstySidecar/` |

## 🔒 Security

- **Read-Only Database Access**: Phase 1 tools only read from the database
- **API Keys Redacted**: Provider queries automatically redact sensitive credentials
- **Local Only**: All operations happen on your local machine
- **No Telemetry**: Zero data collection or external communication

## 🐛 Troubleshooting

### "Msty database not found"

Ensure Msty Studio Desktop has been run at least once to create the database.

### "Sidecar not running"

Start Sidecar from Terminal for best dependency detection:

```bash
open -a MstySidecar
```

### Database locked

If Msty Studio is running, it may have a lock on the database. The MCP uses read-only mode but occasional locks can occur.

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

## 🍍 Credits

Created by **Pineapple 🍍** as part of the AI-Administered Local System project.

Built for seamless integration between Claude Desktop and Msty Studio Desktop.

---

**Current Version**: 2.0.0 (Phase 1)  
**Last Updated**: December 2025  
**Platform**: macOS (Apple Silicon optimized)

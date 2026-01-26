# Changelog

All notable changes to Msty Admin MCP will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [6.0.1] - 2026-01-26

### Fixed
- Enhanced `scan_database_locations` to search more locations and file types
- Now scans for JSON, LevelDB, and other data formats used by Msty 2.4.0+
- Lists all files in directories to help identify where Msty stores data
- Added `found_data_files` to track non-SQLite storage
- Added import for `get_metrics_db_path` to fix prompt template tools

## [6.0.0] - 2026-01-26

### Added
- **Phase 7: Conversation Management** - 3 new tools for working with chat history
  - `export_conversations` - Export conversations in JSON, Markdown, or CSV format
  - `search_conversations` - Search through conversation titles and messages
  - `get_conversation_stats` - Usage statistics and activity patterns

- **Phase 8: Prompt Templates & Automation** - 4 new tools for workflow automation
  - `create_prompt_template` - Create reusable templates with {{variable}} placeholders
  - `list_prompt_templates` - List available templates with built-in examples
  - `run_prompt_template` - Execute templates with variable substitution
  - `smart_model_router` - Intelligent model selection based on task analysis

- **Phase 9: Backup & System Management** - 3 new tools for configuration management
  - `backup_configuration` - Backup personas, prompts, and tools to JSON
  - `restore_configuration` - Restore from backup with dry-run preview
  - `get_system_resources` - CPU, memory, disk usage with model recommendations

- **Total tools now: 42** (up from 32)

### Changed
- Server version bumped to 6.0.0
- Updated docstring to document all 9 phases
- Main entry point now logs all phases

## [5.2.0] - 2026-01-26

### Added
- **Phase 6: Advanced Model Management** - 4 new tools for model insights
  - `get_model_details` - Comprehensive model information including capabilities and use cases
  - `benchmark_model` - Performance benchmarks with tokens/second at different context sizes
  - `list_local_model_files` - Scan disk for MLX and GGUF model files with sizes
  - `estimate_model_requirements` - Estimate RAM/VRAM requirements based on model size and quantization
- **Total tools now: 32**

### Changed
- Server status now includes Phase 6 tools

## [5.1.0] - 2026-01-26

### Added
- **Model Tagging System** - Smart model categorization for intelligent selection
  - `list_model_tags` - View tags for all models or specific model
  - `find_model_by_tag` - Find models by capability (fast, quality, coding, creative, reasoning, etc.)
  - Auto-tagging based on model name patterns (size, purpose, capabilities)
  - Manual overrides for specific models
- **Response Caching** - Reduce redundant API calls
  - `get_cache_stats` - View cache statistics
  - `clear_cache` - Force fresh data fetch
  - Configurable TTL (default 30 seconds)
- **Enhanced Database Detection** - Better support for Msty 2.4.0+ database locations
  - `scan_database_locations` - Find all database files on system
  - Glob pattern search for database files
  - Container/sandbox app support
  - Environment variable override: `MSTY_DATABASE_PATH`
- **4 New Tools** bringing total to 28

### Changed
- `get_server_status` now includes cache stats and database type info
- `get_msty_paths` enhanced with multiple database search strategies
- Server status shows "Phase 5+ - Enhanced"

## [5.0.1] - 2026-01-26

### Fixed
- **Critical**: Chat functions now route to correct service ports
  - `chat_with_local_model` now finds chat models on MLX/LLaMA.cpp ports instead of using embedding models on port 11964
  - `compare_model_responses` now compares chat models from all services
  - `run_calibration_test` now tests chat models instead of embedding models
- **New helper functions**:
  - `get_chat_port_for_model()` - Find which port a specific model is on
  - `get_first_chat_model()` - Find first available chat model (skips embedding models)
- `evaluate_response_heuristic` now returns `passed` field (fixes `evaluate_response_quality` KeyError)
- Embedding models (bge-m3, nomic-embed) are now filtered out when auto-selecting models for chat

## [5.0.0] - 2026-01-25

### Added
- **Full Msty 2.4.0+ Support** - Complete rewrite of service detection for new architecture
  - `MLX_SERVICE_PORT` (11973) - Apple Silicon optimized MLX models
  - `LLAMACPP_SERVICE_PORT` (11454) - LLaMA.cpp GGUF models
  - `VIBE_PROXY_PORT` (8317) - Unified proxy for all AI services
- **New helper functions**:
  - `is_local_ai_available()` - Check service availability by connection (not process)
  - `get_available_service_ports()` - Check all 4 Msty services at once
- **Multi-service model listing** - `list_available_models` now queries ALL services and aggregates results
- **Shell script launcher** - `run_msty_server.sh` for reliable Claude Desktop integration

### Changed
- **BREAKING**: Removed dependency on `MstySidecar` process detection
  - Msty 2.4.0+ has Local AI services built into the main app
  - Now checks service availability by actually connecting to ports
- `get_sidecar_status` completely rewritten:
  - Shows status of all 4 services (Local AI, MLX, LLaMA.cpp, Vibe Proxy)
  - Returns models from the first available service
- `list_available_models` now returns models from ALL services with `by_service` breakdown
- `analyse_msty_health` shows status for all 4 services instead of just Sidecar
- All functions that checked for `MstySidecar` now use `is_local_ai_available()`
- Updated environment variables documentation

### Fixed
- **Critical**: Claude Desktop `cwd` not being respected - added shell script workaround
- Service detection now works with Msty 2.4.0+ architecture
- Models from MLX and LLaMA.cpp services are now visible

### Migration Guide
If upgrading from v4.x:
1. Replace your Claude Desktop config to use the shell script launcher
2. Msty Sidecar is no longer required - services are built into Msty Studio 2.4.0+
3. All existing tools work the same, just with better service detection

## [4.1.0] - 2025-12-30

### Added
- **Environment Variables**: Configurable settings via environment variables
  - `MSTY_SIDECAR_HOST` - Sidecar API host address (default: `127.0.0.1`)
  - `MSTY_AI_PORT` - Local AI Service port (default: `11964`)
  - `MSTY_PROXY_PORT` - Sidecar proxy port (default: `11932`)
  - `MSTY_TIMEOUT` - API request timeout in seconds (default: `10`)
- Enhanced `make_api_request()` with configurable host parameter
- Improved error logging with `logger.warning()` for connection failures and HTTP errors
- Error response body capture (first 200 chars) for better debugging

### Changed
- Updated README with Environment Variables documentation section
- Example `claude_desktop_config.json` now includes env var configuration

### Fixed
- Removed duplicate `LOCAL_AI_SERVICE_PORT` constant from `phase4_5_tools.py`

## [4.0.1] - 2025-12-27

### Fixed
- Removed Acknowledgements section from README for cleaner public release

## [4.0.0] - 2025-12-27

### Added
- **Phase 5: Tiered AI Workflow** - Complete calibration and handoff system
  - `run_calibration_test` - Test local models against standardised prompts
  - `evaluate_response_quality` - Score responses using heuristic evaluation
  - `identify_handoff_triggers` - Track patterns that should escalate to Claude
  - `get_calibration_history` - Historical test results with statistics

- **Phase 4: Intelligence Layer** - Performance analytics and optimisation
  - `get_model_performance_metrics` - Tokens/sec, latency, error rates
  - `analyse_conversation_patterns` - Privacy-respecting usage analytics
  - `compare_model_responses` - Multi-model comparison with quality scoring
  - `optimise_knowledge_stacks` - Knowledge stack analysis and recommendations
  - `suggest_persona_improvements` - AI-powered persona optimisation

- Metrics database (`~/.msty-admin/msty_admin_metrics.db`) for persistent tracking
- Calibration prompts for 5 categories: reasoning, coding, writing, analysis, creative
- Quality scoring rubric with heuristic evaluation

### Changed
- Total tools increased from 15 to 24
- Server version bumped to 4.0.0
- README completely rewritten for public release with:
  - Detailed use cases and examples
  - Comprehensive FAQ section
  - Hardware recommendations table
  - Architecture diagram

## [3.0.0] - 2025-12-26

### Added
- **Phase 3: Automation Bridge** - Local model integration via Sidecar API
  - `get_sidecar_status` - Sidecar and Local AI Service health check
  - `list_available_models` - Query available models via Ollama-compatible API
  - `query_local_ai_service` - Direct low-level API access
  - `chat_with_local_model` - Send messages with automatic metric tracking
  - `recommend_model` - Hardware-aware model recommendations

### Changed
- Total tools increased from 10 to 15
- Added psutil dependency for process monitoring

## [2.0.0] - 2025-12-25

### Added
- **Phase 2: Configuration Management** - Sync and export tools
  - `export_tool_config` - Export MCP configurations for backup
  - `import_tool_config` - Validate and prepare tools for Msty import
  - `sync_claude_preferences` - Convert Claude preferences to Msty persona
  - `generate_persona` - Create personas from templates (opus, minimal, coder, writer)

### Changed
- Total tools increased from 6 to 10

## [1.0.0] - 2025-12-24

### Added
- **Phase 1: Foundational Tools** - Read-only database access
  - `detect_msty_installation` - Detect Msty Studio paths and status
  - `read_msty_database` - Query conversations, personas, prompts, tools
  - `list_configured_tools` - View MCP toolbox configuration
  - `get_model_providers` - List configured AI providers
  - `analyse_msty_health` - Database integrity and storage analysis
  - `get_server_status` - MCP server info and capabilities

- Initial project structure with FastMCP server
- MIT License
- Basic README documentation

---

## Version History Summary

| Version | Date | Phase | Tools |
|---------|------|-------|-------|
| 6.0.0 | 2026-01-26 | Phases 7-9 | 42 |
| 5.2.0 | 2026-01-26 | Model Management | 32 |
| 5.1.0 | 2026-01-26 | Tagging + Caching | 28 |
| 5.0.1 | 2026-01-26 | Bugfix | 24 |
| 5.0.0 | 2026-01-25 | Msty 2.4.0+ | 24 |
| 4.1.0 | 2025-12-30 | Enhancement | 24 |
| 4.0.1 | 2025-12-27 | Bugfix | 24 |
| 4.0.0 | 2025-12-27 | Phase 5 | 24 |
| 3.0.0 | 2025-12-26 | Phase 3 | 15 |
| 2.0.0 | 2025-12-25 | Phase 2 | 10 |
| 1.0.0 | 2025-12-24 | Phase 1 | 6 |

[6.0.0]: https://github.com/DRVBSS/msty-admin-mcp/compare/v5.2.0...v6.0.0
[5.2.0]: https://github.com/DRVBSS/msty-admin-mcp/compare/v5.1.0...v5.2.0
[5.1.0]: https://github.com/DRVBSS/msty-admin-mcp/compare/v5.0.1...v5.1.0
[5.0.1]: https://github.com/DRVBSS/msty-admin-mcp/compare/v5.0.0...v5.0.1
[5.0.0]: https://github.com/DRVBSS/msty-admin-mcp/compare/v4.1.0...v5.0.0
[4.1.0]: https://github.com/M-Pineapple/msty-admin-mcp/compare/v4.0.1...v4.1.0
[4.0.1]: https://github.com/M-Pineapple/msty-admin-mcp/compare/v4.0.0...v4.0.1
[4.0.0]: https://github.com/M-Pineapple/msty-admin-mcp/compare/v3.0.0...v4.0.0
[3.0.0]: https://github.com/M-Pineapple/msty-admin-mcp/compare/v2.0.0...v3.0.0
[2.0.0]: https://github.com/M-Pineapple/msty-admin-mcp/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/M-Pineapple/msty-admin-mcp/releases/tag/v1.0.0

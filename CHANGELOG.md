# Changelog

All notable changes to Msty Admin MCP will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [6.6.0] - 2026-01-26

### Added
- **Enhanced Tagging System v2.0** - Complete rewrite with improved intelligence
  - Context length awareness with `long_context` (100K+), `very_long_context` (250K+), `massive_context` (500K+)
  - Quantization detection: `fp16`, `8bit`, `6bit`, `5bit`, `4bit`, `3bit`
  - Architecture tags: `moe` (Mixture of Experts), `mlx`, `gguf`
  - New size tier: `massive` (200B+ parameters)
  - New capability tags: `thinking`, `math`, `science`, `instruct`, `chat`
- `TAG_DEFINITIONS` constant with human-readable descriptions for all tags
- `get_all_tags()` function to retrieve tag definitions
- `detect_model_size()` function for parameter count extraction
- `detect_context_length()` function with known model database
- Improved pattern matching with negative lookaheads (e.g., `mini` but not `ministral`)
- 61 unit tests for tagging module (all passing)

### Changed
- Size detection now properly handles 3-digit parameter counts (405B, 253B, 235B)
- MoE models correctly detected via `A\d+B` pattern (e.g., A22B for active parameters)
- Context tags are now cumulative (massive_context includes very_long_context includes long_context)
- Better regex patterns to avoid false positives (`pro` but not `proxy`, `dev` but not `device`)

## [6.5.0] - 2026-01-26

### Added
- **Complete Modular Architecture** - Extracted all utility functions into dedicated modules
  - `src/paths.py` - Path resolution utilities (get_msty_paths, sanitize_path, expand_path, read_claude_desktop_config)
  - `src/database.py` - Database utilities with SQL injection protection (safe_query_table, safe_count_table, validate_table_exists)
  - `src/network.py` - API request helpers (make_api_request, is_process_running, is_local_ai_available, get_available_service_ports)
  - `src/cache.py` - Response caching (ResponseCache class, get_cached_models, cache_models, get_cache)
  - `src/tagging.py` - Model tagging system (MODEL_TAGS, get_model_tags, find_models_by_tag)
- All modules have proper `__all__` exports for clean imports
- Updated `src/__init__.py` with exports for all new modules

### Changed
- Server.py reduced from 3575 to ~3045 lines (~15% total reduction from v6.0.0)
- Improved separation of concerns - each module has a single responsibility
- Better testability - utility functions can now be unit tested independently
- Cleaner imports - server.py now imports from dedicated modules

## [6.4.0] - 2026-01-26

### Added
- **Modular Architecture** - Extracted core components into separate modules
  - `src/constants.py` - Server version, port configs, table allowlists, model patterns
  - `src/models.py` - Data classes (MstyInstallation, MstyHealthReport, DatabaseStats, PersonaConfig)
  - `src/errors.py` - ErrorCode class and response helpers
- Updated `src/__init__.py` with proper exports for all modules

### Changed
- Server.py reduced from 3741 to 3575 lines (~4.5% reduction)
- Version now sourced from `constants.py` for single source of truth
- Improved import organization in server.py

## [6.3.0] - 2026-01-26

### Added
- **Standardized Error Response System** - Consistent API error format across all tools
  - `ErrorCode` class with standard codes (DATABASE_NOT_FOUND, SERVICE_UNAVAILABLE, etc.)
  - `error_response()` helper for creating standardized JSON error responses
  - `success_response()` helper for standardized success responses
  - `make_error_response()` and `make_success_response()` dict helpers
- All error responses now include: `success`, `error.code`, `error.message`, `error.suggestion`, `timestamp`

### Changed
- Updated key tools to use standardized error responses:
  - `read_msty_database` - now returns structured error with suggestion
  - `list_configured_tools` - consistent error format
  - `query_local_ai_service` - SERVICE_UNAVAILABLE with suggestion
  - `recommend_model` - INVALID_PARAMETER with valid options
  - `export_conversations` - DATABASE_NOT_FOUND with suggestion

## [6.2.0] - 2026-01-26

### Fixed
- **SQL Injection Protection** - Added comprehensive protection against SQL injection attacks
  - `ALLOWED_TABLE_NAMES` frozenset with safe table names
  - `is_safe_table_name()` function validates tables against allowlist
  - `validate_table_exists()` verifies table exists AND is safe to query
  - `safe_query_table()` and `safe_count_table()` helper functions
  - All database operations now use parameterized queries
- **Removed all bare `except:` clauses** - Now use specific exception types:
  - `sqlite3.Error` for database operations
  - `OSError`, `PermissionError` for file operations
  - `psutil.NoSuchProcess`, `psutil.AccessDenied` for process operations
  - `urllib.error.URLError`, `urllib.error.HTTPError` for network operations
- **Fixed shell script portability** - `run_msty_server.sh` now uses `SCRIPT_DIR` instead of hardcoded path
- **Removed duplicate function definitions** - Eliminated 6 duplicate Phase 7/8/9 tool functions

### Changed
- Server version bumped to 6.2.0
- All 42 tools now unique (was incorrectly counting duplicates)

## [6.1.0] - 2026-01-26

### Added
- **Comprehensive Model Tagging** - 60+ model-specific tag overrides for accurate routing
- New pattern matches: `flash`, `lite`, `haiku`, `pro`, `ultra`, `nemotron`, `v3`, `dev`, `kimi-dev`, `oswe`, `grok-code`, `nemo`, `mistral-nemo`, `jina-embed`, `pro-image`
- All Claude models now properly tagged with `coding`, `reasoning`, `creative`
- All GPT models properly tagged with capabilities
- All Gemini models properly tagged
- Hybrid models (gemini-claude-*) properly tagged

### Fixed
- **Size detection order** - Now checks large sizes first to avoid false `small` matches on 253B models
- Fixed Qwen3-235B incorrectly tagged as `small` (now `large`)
- Fixed Nemotron-Ultra-253B incorrectly tagged as `small` (now `large`)
- Fixed Mistral-Nemo missing `long_context` tag (has 1M context!)
- Fixed Kimi-Dev-72B missing `coding` tag
- Fixed Hermes-70B incorrectly tagged as `small` (now `large`)

## [6.0.2] - 2026-01-26

### Fixed
- **Database detection now finds SharedStorage** - Msty 2.4.0+ stores data in `SharedStorage` file (no .db extension)
- Added explicit check for `SharedStorage` in main data directory
- Database should now be auto-detected without environment variable

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

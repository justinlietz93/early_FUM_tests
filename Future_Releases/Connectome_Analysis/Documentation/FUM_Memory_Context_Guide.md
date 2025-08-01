# FUM Memory System Context Guide

## Overview
This document provides instructions for accessing the complete context of the FUM debugging project stored in the MCP memory system.

## Memory System Location
The MCP memory server data is managed by `@modelcontextprotocol/server-memory`. Storage location depends on configuration:
- Check MCP server startup parameters for storage flags
- Look in `server-memory/dist/index.js` for configuration details
- May be in-memory, file-based, or database-backed storage

## Key Memory Query Commands

### 1. Get Complete Project Status
```javascript
search_nodes with query: "FUM debugging project status"
// OR
open_nodes with: ["FUM_Current_Project_State"]
```

### 2. Get GDSP Repair Implementation Details
```javascript
search_nodes with query: "GDSP homeostatic repair implementation"
// OR
open_nodes with: ["GDSP_Homeostatic_Repair_Fix"]
```

### 3. Get Testing/Validation Strategy
```javascript
search_nodes with query: "FUM testing validation criteria"
// OR
open_nodes with: ["FUM_Testing_Strategy"]
```

### 4. Get Future Architecture Plans
```javascript
search_nodes with query: "nexus unified interface cybernetic"
// OR
open_nodes with: ["FUM_Architecture_Vision"]
```

### 5. Get All Technical Implementations
```javascript
search_nodes with query: "code_implementation"
```

### 6. Get Complete Knowledge Graph
```javascript
read_graph
```

## Current Project State Summary

### Status: GDSP Homeostatic Repair Fix Implemented, Awaiting Validation

### Critical Issue Resolved:
- **Problem**: GDSP `trigger_homeostatic_repairs()` function only made single repair attempt per cycle
- **Root Cause**: Single `return _grow_connection_across_gap(substrate)` statement caused immediate exit
- **Impact**: With 4 disconnected components, minimum 3 bridging connections needed - single attempt insufficient
- **Solution**: Implemented iterative healing approach in `fum/mechanisms/gdsp.py` lines 172-200

### Technical Fix Details:
- **Iterative Logic**: Up to 10 repair attempts per cycle with `max_healing_attempts = min(10, initial_component_count * 2)`
- **Debug Logging**: Comprehensive tracking with `[GDSP HOMEOSTATIC DEBUG]` prefix
- **Progress Monitoring**: Tracks connection counts before/after each attempt
- **Termination**: Stops when no more connections can be added or max attempts reached

### Next Steps Required:
1. **Test GDSP Fix**: Run `fum/cultivation/germinate.py` to validate homeostatic repair
2. **Validation Criteria**: 
   - `component_count` should reduce from 4.0 → 1.0
   - `connectome_entropy` should decrease from 1.593494
   - `repair_triggered` should become False after healing
   - Debug logs should show multiple healing attempts with connection additions

### Future Architecture Vision:
- **Goal**: Make nexus the unified interface for all external interactions
- **Philosophy**: Create true cybernetic organism where external systems only interface with nexus
- **Timeline**: Implement after Phase 1 functions correctly

## File Locations:
- **Primary Fix**: `fum/mechanisms/gdsp.py` (lines 172-200)
- **Test Script**: `fum/cultivation/germinate.py`
- **Previous Evidence**: `runs/cultivated_seed_1753890486/console_log.txt`
- **Blueprint**: `FullyUnifiedModel/DO_NOT_DELETE/FUM_Blueprint_DO_NOT_DELETE.md`

## Memory System Transport Instructions:
1. Stop MCP server gracefully to ensure data persistence
2. Locate storage files (check server configuration)
3. Copy storage files to new location
4. Restart MCP server with same configuration in new environment
5. Verify knowledge graph integrity with `read_graph` command

---
*Generated: 2025-07-31T01:55:00Z*
*Context saved to MCP memory system with entities: FUM_Current_Project_State, GDSP_Homeostatic_Repair_Fix, FUM_Testing_Strategy, FUM_Architecture_Vision, Memory_Query_Instructions*
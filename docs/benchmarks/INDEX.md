# OneRec Benchmark Documentation Index

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../assets/vLLM-GR.png">
    <img alt="vLLM" src="vLLM-GR.png" width=55%>
  </picture>
</p>


Complete documentation suite for the OneRec generative recommendation benchmark.

## 📚 Documentation Overview

### For Getting Started

1. **[README.md](README.md)** ⭐ **START HERE**
   - Quick start guide
   - Installation instructions
   - Basic usage examples
   - Overview of 8 task types
   - Command-line arguments reference

### For Understanding Usage

2. **[OPERATION_MODES.md](OPERATION_MODES.md)** 
   - Two operation modes explained (Throughput and API Server)
   - Step-by-step CLI commands for each mode
   - Mode comparison and decision matrix
   - launch.json configuration examples
   - Results monitoring and analysis

3. **[VLLM_EVALUATION.md](VLLM_EVALUATION.md)** 
   - Using vLLM for LLM-based evaluation
   - Setting up evaluation server
   - Configuration for vLLM endpoints
   - Async/sync client usage
   - Advanced server configurations

4. **[CONFIGURATION.md](CONFIGURATION.md)**
   - Detailed task configurations
   - Generation parameter explanations
   - Evaluation metrics reference
   - Custom configuration examples
   - Configuration best practices

4. **[CONFIGURATION.md](CONFIGURATION.md)**
   - Detailed task configurations
   - Generation parameter explanations
   - Evaluation metrics reference
   - Custom configuration examples
   - Configuration best practices

5. **[DATA_FORMAT.md](DATA_FORMAT.md)**
   - Data directory structure
   - Parquet schema specification
   - Message and metadata formats
   - Data examples for each task
   - Data validation scripts
   - Sample data creation methods

### For Development

6. **[API_REFERENCE.md](API_REFERENCE.md)**
   - Python API documentation
   - OneRecDataset class reference
   - BaseLoader class reference
   - TaskRegistry reference
   - Evaluators overview
   - Type hints and examples

7. **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)**
   - Common issues and solutions
   - Diagnostic scripts
   - Debug mode instructions
   - Getting help guidelines

## 🎯 Quick Navigation by Task

### If you want to...

**Run a benchmark**
→ [OPERATION_MODES.md](OPERATION_MODES.md) - Choose your mode and get CLI commands

**Choose between Throughput vs API Server mode**
→ [OPERATION_MODES.md - Mode Comparison](OPERATION_MODES.md#mode-comparison)

**Configure generation parameters**
→ [CONFIGURATION.md - Generation Parameters](CONFIGURATION.md#generation-parameters)

**Prepare your data**
→ [DATA_FORMAT.md - Creating Sample Data](DATA_FORMAT.md#creating-sample-data)

**Understand task specifics**
→ [README.md - Task Types](README.md#task-types)

**Use Python API**
→ [API_REFERENCE.md - OneRecDataset](API_REFERENCE.md#onerecdata set)

**Fix an error**
→ [TROUBLESHOOTING.md - Common Issues](TROUBLESHOOTING.md#common-issues--solutions)

## 📖 Documentation Structure

### README.md (5-10 min read)
```
├── Overview (tasks, features)
├── Quick Start (basic usage)
├── Installation (setup)
├── Usage (command-line args, examples)
├── Configuration (task configs)
├── Data Format (overview)
├── Task Types (descriptions)
├── Evaluation (metrics)
└── Troubleshooting (common issues)
```

### CONFIGURATION.md (10-15 min read)
```
├── Task Configurations (detailed)
├── Generation Parameters (sampling, output control)
├── Evaluation Metrics (supported metrics)
├── Custom Configuration (modification guide)
└── Configuration Examples (presets)
```

### DATA_FORMAT.md (10-15 min read)
```
├── Directory Structure (file organization)
├── File Format (Parquet)
├── Schema Definition (messages, metadata)
├── Data Examples (task-specific examples)
├── Data Validation (validation scripts)
└── Creating Sample Data (generation methods)
```

### API_REFERENCE.md (15-20 min read)
```
├── OneRecDataset (class reference)
├── BaseLoader (class reference)
├── TaskRegistry (configuration access)
├── Evaluators (evaluation classes)
├── Utilities (helper functions)
└── Complete Example (full workflow)
```

### TROUBLESHOOTING.md (5-20 min read)
```
├── Quick Diagnostics (setup check)
├── Common Issues & Solutions (organized by category)
├── Debug Mode (how to debug)
└── Getting Help (info to provide)
```

## 🚀 Getting Started Workflows

### Workflow 1: Run Benchmark in 5 Minutes

1. **Install**: [README.md - Installation](README.md#installation)
2. **Prepare Data**: [DATA_FORMAT.md - Creating Sample Data](DATA_FORMAT.md#creating-sample-data)
3. **Run**: [README.md - Quick Start](README.md#quick-start)

### Workflow 2: Understand Tasks in Depth

1. **Read Overview**: [README.md - Task Types](README.md#task-types)
2. **Check Config**: [CONFIGURATION.md - Task Configurations](CONFIGURATION.md#task-configurations)
3. **View Examples**: [DATA_FORMAT.md - Data Examples](DATA_FORMAT.md#data-examples)

### Workflow 3: Customize Configuration

1. **Understand Params**: [CONFIGURATION.md - Generation Parameters](CONFIGURATION.md#generation-parameters)
2. **View Examples**: [CONFIGURATION.md - Configuration Examples](CONFIGURATION.md#configuration-examples)
3. **Apply Changes**: [CONFIGURATION.md - Custom Configuration](CONFIGURATION.md#custom-configuration)

### Workflow 4: Integrate into Python Code

1. **Learn API**: [API_REFERENCE.md - OneRecDataset](API_REFERENCE.md#onerecdata set)
2. **Check Examples**: [API_REFERENCE.md - Complete Example Workflow](API_REFERENCE.md#complete-example-workflow)
3. **Debug Issues**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

## 📋 Key Sections by Document

### README.md Sections
- Installation with prerequisites
- Installation steps
- Operation modes overview with examples
- 4 command-line example workflows
- All command-line arguments table
- 8 detailed task type descriptions
- Evaluation workflow
- Common troubleshooting (defer to TROUBLESHOOTING.md for details)

### OPERATION_MODES.md Sections 
- Overview of two modes (Throughput vs API Server)
- Step-by-step instructions for each mode
- Mode comparison table and decision matrix
- Command reference with examples
- launch.json configuration
- Results monitoring
- Troubleshooting per mode
### VLLM_EVALUATION.md Sections ⭐ **NEW**
- Overview of vLLM for evaluation
- Setup and installation steps
- Configuration options for vLLM endpoints
- Code integration examples
- Async/sync client usage
- Advanced server configurations
- Monitoring and troubleshooting
### CONFIGURATION.md Sections
- Complete configuration schema
- 8 task configuration examples with explanations
- Common generation parameters with presets
- 9 supported metrics with ranges
- How to modify and create configurations
- 5 configuration examples (fast baseline, high-quality, etc.)

### DATA_FORMAT.md Sections
- Directory structure specification
- Parquet format advantages
- Schema definition with field descriptions
- 5 data examples for different tasks
- Validation checklist and scripts
- 4 methods to create sample data
- Data statistics and privacy notes
- Troubleshooting for data issues

### API_REFERENCE.md Sections
- OneRecDataset constructor, methods, properties
- BaseLoader constructor, methods
- TaskRegistry access and modification
- 4 evaluator classes
- Utility functions
- Complete workflow example
- Error handling patterns
- Type hints reference

### TROUBLESHOOTING.md Sections
- Diagnostic script
- 8 categories of issues with solutions
- Specific error messages and fixes
- Debug mode instructions
- Component testing
- How to report issues
- Useful debugging commands

## 🎓 Learning Paths

### Path 1: Quick User (30 minutes)
1. README.md - Quick Start (5 min)
2. README.md - Command-line Arguments (5 min)
3. DATA_FORMAT.md - Creating Sample Data (5 min)
4. README.md - Usage Examples (10 min)
5. Run first benchmark! (5 min)

### Path 2: Power User (2 hours)
1. Complete README.md (30 min)
2. CONFIGURATION.md - Full guide (45 min)
3. DATA_FORMAT.md - Full guide (30 min)
4. Try different configurations (15 min)

### Path 3: Developer (3 hours)
1. Complete README.md (30 min)
2. API_REFERENCE.md - Full reference (60 min)
3. Write Python integration code (30 min)
4. TROUBLESHOOTING.md - Debug tips (30 min)
5. Integrate into your system (30 min)

### Path 4: Troubleshooter (1 hour)
1. TROUBLESHOOTING.md - Quick Diagnostics (10 min)
2. Run diagnostic script (5 min)
3. Find your issue (10 min)
4. Apply solution (10 min)
5. Verify fix (5 min)

## 📊 Document Statistics

| Document | Pages | Est. Read Time | Best For |
|----------|-------|----------------|----------|
| README.md | 8 | 10 min | Quick start, overview |
| OPERATION_MODES.md | 10 | 15 min | Running benchmarks |
| VLLM_EVALUATION.md | 8 | 10 min | LLM-based evaluation |
| CONFIGURATION.md | 12 | 15 min | Configuration, parameters |
| DATA_FORMAT.md | 15 | 15 min | Data preparation |
| API_REFERENCE.md | 10 | 20 min | Python integration |
| TROUBLESHOOTING.md | 12 | 15 min | Problem solving |
| **Total** | **75** | **100 min** | Complete understanding |

## 🔗 Cross-References

### From README.md
- See [CONFIGURATION.md](CONFIGURATION.md) for detailed task configuration
- See [DATA_FORMAT.md](DATA_FORMAT.md) for data format specification
- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for error solutions

### From CONFIGURATION.md
- See [README.md](README.md#usage) for usage examples
- See [DATA_FORMAT.md](DATA_FORMAT.md) for data structure
- See [API_REFERENCE.md](API_REFERENCE.md) for task registry access

### From DATA_FORMAT.md
- See [README.md](README.md#quick-start) for quick start
- See [API_REFERENCE.md](API_REFERENCE.md) for Python API
- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#2-data-related-issues) for data issues

### From API_REFERENCE.md
- See [README.md](README.md) for overview
- See [CONFIGURATION.md](CONFIGURATION.md) for task configurations
- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for debugging

### From TROUBLESHOOTING.md
- See [README.md](README.md) for basic setup
- See [CONFIGURATION.md](CONFIGURATION.md) for configuration validation
- See [DATA_FORMAT.md](DATA_FORMAT.md) for data validation

## ✅ Documentation Checklist

- [x] README.md - Quick start and usage guide
- [x] CONFIGURATION.md - Detailed configuration reference
- [x] DATA_FORMAT.md - Data format specification
- [x] API_REFERENCE.md - Python API documentation
- [x] TROUBLESHOOTING.md - Problem-solving guide
- [x] This INDEX - Navigation and overview

## 🎯 Next Steps

1. **New to OneRec?**
   → Start with [README.md](README.md)

2. **Need to set up data?**
   → Go to [DATA_FORMAT.md](DATA_FORMAT.md#creating-sample-data)

3. **Integrating with Python?**
   → Read [API_REFERENCE.md](API_REFERENCE.md)

4. **Experiencing issues?**
   → Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

5. **Want advanced configuration?**
   → See [CONFIGURATION.md](CONFIGURATION.md)

## 📞 Support Resources

- **Documentation**: All files in this directory
- **Issues**: [GitHub Issues](https://github.com/JiusiServe/vllm-gr/issues)
- **Diagnostics**: Run diagnostic script in [TROUBLESHOOTING.md](TROUBLESHOOTING.md#quick-diagnostics)
- **Examples**: [README.md - Usage](README.md#usage) and [API_REFERENCE.md - Complete Example](API_REFERENCE.md#complete-example-workflow)

---

**Last Updated**: January 20, 2026  
**Documentation Version**: 1.0  
**OneRec Version**: v1.0

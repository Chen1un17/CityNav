# Multi-Agent Traffic Control System

## Project Overview

This project implements an advanced **Multi-Agent Traffic Control System** that coordinates Regional Agents, Traffic Agent, and Prediction Engine for optimal traffic management using Large Language Models (LLMs). 

**Key Innovation**: Transform traditional single-vehicle route planning into a coordinated multi-agent architecture where intelligent agents collaborate to minimize travel time and optimize traffic flow across urban networks.

### 🚗 System Architecture

- **Regional Agents** (20 agents): Manage intra-regional routing, lane assignment, and green wave coordination
- **Traffic Agent** (1 agent): Handles macro-level inter-regional routing and traffic flow management  
- **Prediction Engine**: Provides traffic flow forecasting using autoregressive models with multiple time windows
- **Comprehensive Logging**: Real-time tracking of all decisions, performance metrics, and system behavior

This extends the original USTBench-Route-Planning with sophisticated multi-agent coordination and predictive capabilities.

## Installation (Python 3.10)

```bash
# Step 1: Install PyTorch and related packages
pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1

# Step 2: Install other dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Check System Requirements
```bash
python run_multi_agent.py check
```

### 2. Run Quick Test (30 minutes simulation)
```bash
python run_multi_agent.py quick
```

### 3. Run Full Multi-Agent Simulation (12 hours)
```bash
python run_multi_agent.py full
```

### 4. Compare Single vs Multi-Agent Performance
```bash
python run_multi_agent.py compare
```

## 📊 Usage Examples

### Multi-Agent Traffic Control (New Default)
```bash
python main.py --multi-agent --location Manhattan --step-size 180 --max-steps 43200
```

### Traditional Single-Agent Baseline
```bash
python main.py --single-agent --location Manhattan --step-size 180 --max-steps 43200
```

### Custom Configuration
```bash
python run_multi_agent.py custom \
    --llm "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --batch-size 16 \
    --location Manhattan \
    --step-size 120 \
    --max-steps 21600
```

## ⚙️ Parameter Descriptions

| Parameter | Description | Default | Multi-Agent Feature |
|-----------|-------------|---------|-------------------|
| `--llm-path-or-name` | LLM model identifier | `meta-llama/Meta-Llama-3.1-8B-Instruct` | Used by all agents |
| `--batch-size` | LLM batch processing size | 16 | Optimized for regional batching |
| `--location` | Simulation location | Manhattan | Region-aware processing |
| `--multi-agent` | **Use multi-agent architecture** | **True** | **🆕 Enable coordinated agents** |
| `--single-agent` | Use traditional baseline | False | Disable multi-agent features |
| `--no-reflection` | Disable LLM reflection | False | Affects all agent decisions |
| `--step-size` | Decision interval (seconds) | 180.0 | Agent coordination timing |
| `--max-steps` | Maximum simulation steps | 43200 | Extended for complex scenarios |

## 🏗️ Multi-Agent Architecture Details

### Regional Agent (20 instances)
- **Purpose**: Manages traffic within individual regions
- **Key Functions**:
  - Plans optimal paths to boundary edges considering traffic conditions
  - Assigns lanes to vehicles to minimize congestion
  - Coordinates green wave traffic signal timing
  - Batch processes multiple vehicle decisions using LLM for efficiency
- **Decision Interval**: 30 seconds
- **LLM Integration**: Context-aware prompts with real-time traffic data

### Traffic Agent (1 instance)
- **Purpose**: Macro-level traffic coordination across regions
- **Key Functions**:
  - Plans region-to-region routes for vehicles (e.g., Region2→Region34)
  - Monitors and manages cutting edge congestion between regions
  - Provides recommendations to Regional Agents
  - Updates macro routes dynamically based on real-time conditions
- **Update Interval**: 60 seconds
- **LLM Integration**: Complex routing scenarios use LLM with comprehensive regional data

### Prediction Engine
- **Purpose**: Traffic flow forecasting and bottleneck prediction
- **Key Functions**:
  - Autoregressive prediction with multiple time windows (180s, 360s, 540s, 720s)
  - Real-time model training and adaptation
  - Congestion forecasting and bottleneck detection
  - Provides predictions to both Regional and Traffic Agents
- **Update Interval**: 30 seconds
- **Models**: Ridge regression with engineered features

### Logger System
- **Real-time Progress**: Console progress bars showing key metrics
- **Comprehensive Logging**: Vehicle tracking, agent decisions, LLM performance
- **Performance Analytics**: Success rates, travel times, throughput analysis
- **Automated Reports**: JSON summary reports with detailed metrics

## 📈 Performance Monitoring

### Real-time Console Output
```
[████████████████████████████████] 75.3% | Vehicles: 1506/2000 | Avg Travel Time: 342.1s | 
Decisions: 4521/4683 (96.5%) | LLM Calls: 157 (Avg: 2.34s) | Runtime: 1847s
```

### Generated Log Files
- `vehicles_YYYYMMDD_HHMMSS.jsonl`: Vehicle tracking and completion
- `regional_decisions_YYYYMMDD_HHMMSS.jsonl`: Regional agent decisions
- `traffic_decisions_YYYYMMDD_HHMMSS.jsonl`: Traffic agent macro routing
- `llm_calls_YYYYMMDD_HHMMSS.jsonl`: LLM API call performance
- `performance_YYYYMMDD_HHMMSS.jsonl`: System performance metrics
- `summary_report_YYYYMMDD_HHMMSS.json`: Final simulation summary

## 🔧 System Requirements

### Software Dependencies
- Python 3.8+
- SUMO Traffic Simulator
- Required packages: `numpy`, `networkx`, `scikit-learn`, `pandas`, `tqdm`

### Data Requirements (Enhanced)
- Road network data (SUMO format)
- **Region partition data** (boundary edges, edge-to-region mapping) - New!
- Vehicle route data (SUMO route format)
- Task configuration (JSON format)

### Hardware Recommendations
- **CPU**: Multi-core processor (4+ cores recommended for parallel agent processing)
- **RAM**: 8GB+ (16GB recommended for 20 regional agents)
- **Storage**: 2GB+ for comprehensive log files
- **GPU**: Optional, for LLM acceleration

## 🔍 Key Improvements Over Single-Agent

1. **Scalability**: 20 Regional Agents handle traffic in parallel
2. **Coordination**: Traffic Agent provides macro-level optimization
3. **Prediction**: Autoregressive models forecast traffic patterns
4. **Efficiency**: Batch processing reduces LLM API calls
5. **Monitoring**: Comprehensive logging and real-time progress tracking
6. **Adaptability**: Dynamic route updates based on changing conditions

## 📊 Expected Performance Gains

- **Travel Time Reduction**: 15-25% improvement in average travel time
- **Throughput Increase**: 10-20% more vehicles completing routes
- **Decision Quality**: Higher success rate due to coordinated planning
- **System Efficiency**: Reduced LLM calls through intelligent batching

## Legacy Support

The original single-agent route planning is still available:
```bash
python main.py --single-agent
```

This maintains compatibility with the original USTBench-Route-Planning implementation while providing the new multi-agent capabilities as the default mode.

# Case Studies for Decentralized Multi-Agent Reinforcement Learning in Local Energy Markets

This directory contains six comprehensive case studies designed to systematically answer the main research question of this doctoral research project.

## Research Question

> **Main:** How does implicit cooperation, enabled by MARL, improve the management of DERs to maximize their energy use while ensuring the balance between supply and demand in LEMs?

> **Complementary:** How do different market mechanism designs and incentive structures influence agent participation strategies, energy allocation decisions and grid operations in decentralized local energy markets to maintain grid balance while maximizing economic efficiency, and how can agent-based modeling frameworks evaluate these interactions while considering technical constraints and uncertainties?

## Case Studies Overview

### Case 1: Market Mechanism Comparison (`case1_market_mechanisms.py`)

**Objective**: Compare different clearing mechanisms to identify optimal pricing approaches for decentralized energy markets.

- **Scenarios**: 6 scenarios testing different clearing mechanisms (AVERAGE, BUYER, SELLER, BID_ASK_SPREAD, NASH_BARGAINING, PROPORTIONAL_SURPLUS)
- **Agents**: 6 diverse agents with different capacities and profiles
- **Key Variables**: Market clearing mechanism
- **Controlled Factors**: Agent configurations, grid topology, DSO policies
- **Expected Insights**: Which mechanisms promote better coordination and market efficiency

### Case 2: Agent Heterogeneity & Market Power (`case2_agent_heterogeneity.py`)

**Objective**: Analyze how agent size differences and market concentration affect coordination and strategic behavior.

- **Scenarios**: 4 market structures (balanced, monopoly, oligopoly, cooperative)
- **Agents**: Variable (6-7 agents per scenario)
- **Key Variables**: Agent capacity distribution, market concentration
- **Controlled Factors**: Market mechanism, grid configuration, DSO policies
- **Expected Insights**: Impact of market power on coordination effectiveness and agent strategies

### Case 3: DSO Intervention Strategies (`case3_dso_intervention.py`)

**Objective**: Study how different DSO regulatory approaches affect market participation and grid stability.

- **Scenarios**: 4 regulatory approaches (permissive, moderate, strict, dynamic)
- **Agents**: 8 agents with diverse generation/demand profiles
- **Key Variables**: DSO intervention thresholds, penalty structures
- **Controlled Factors**: Agent configurations, market mechanism, grid topology
- **Expected Insights**: Optimal balance between market freedom and grid stability

### Case 4: Grid Topology & Congestion Effects (`case4_grid_constraints.py`)

**Objective**: Examine how grid physical constraints and topologies affect market outcomes and coordination.

- **Scenarios**: 12 combinations of 4 topologies × 3 capacity levels
- **Agents**: 9 geographically distributed agents
- **Key Variables**: Grid topology (STAR, MESH, RING, TREE), capacity constraints
- **Controlled Factors**: Agent configurations, market mechanism, DSO policies
- **Expected Insights**: Grid topology effects on market efficiency and constraint management

### Case 5: Battery Storage Coordination (`case5_battery_coordination.py`)

**Objective**: Analyze how different storage configurations enhance implicit coordination and market efficiency.

- **Scenarios**: 6 storage deployment strategies
- **Agents**: 7 agents with varying battery configurations
- **Key Variables**: Battery deployment, capacity ratios, efficiency levels
- **Controlled Factors**: Agent generation/demand, market mechanism, grid configuration
- **Expected Insights**: Role of storage in coordination and optimal deployment strategies

### Case 6: Implicit Cooperation Effectiveness (`case6_implicit_cooperation.py`)

**Objective**: Directly test how implicit cooperation improves DER management and supply-demand balance (main research question).

- **Scenarios**: 6 scenarios testing coordination effectiveness
- **Agents**: 8 agents with complementary profiles designed for coordination
- **Key Variables**: Coordination level (none, implicit, explicit), cooperation conditions
- **Controlled Factors**: Agent configurations, market mechanism, grid topology
- **Expected Insights**: Quantitative validation of implicit cooperation effectiveness in LEMs

## Framework Structure

```
cases/
├── __init__.py                    # Package initialization and utilities
├── case1_market_mechanisms.py     # Market mechanism comparison
├── case2_agent_heterogeneity.py   # Agent heterogeneity analysis
├── case3_dso_intervention.py      # DSO intervention strategies
├── case4_grid_constraints.py      # Grid capacity constraints
├── case5_battery_coordination.py  # Battery storage coordination
├── case6_implicit_cooperation.py  # Implicit cooperation effectiveness
├── case_studies_overview.py       # Comprehensive analysis framework
└── README.md                      # This documentation
```

## Research Dimensions Coverage

| Research Dimension                 | Primary Cases | Secondary Cases | Total Coverage |
| ---------------------------------- | ------------- | --------------- | -------------- |
| Market Mechanism Design            | Case 1        | Cases 2, 3      | High           |
| Incentive Structures               | Case 3        | Cases 1, 2      | High           |
| Agent Participation Strategies     | Case 2        | Cases 1, 5      | High           |
| Energy Allocation Decisions        | Case 5        | Cases 2, 4      | High           |
| Grid Operations                    | Case 4        | Cases 3, 5      | High           |
| Technical Constraints              | Cases 4, 5    | Case 3          | High           |
| Implicit Cooperation Effectiveness | Case 6        | Cases 1, 5      | High           |

## Usage

### Import and Run All Case Studies

```python
from cases import get_all_case_studies, print_case_studies_summary

# Get summary of all case studies
print_case_studies_summary()

# Get all scenarios from all case studies
all_scenarios = get_all_case_studies()
print(f"Total scenarios: {len(all_scenarios)}")
```

### Run Specific Case Study

```python
from cases import get_case_study

# Get scenarios from Case 1
case1_scenarios = get_case_study("case1_market_mechanisms")

# Iterate through scenarios
for scenario_name, config in case1_scenarios.items():
    print(f"Scenario: {scenario_name}")
    print(f"Agents: {len(config.agents)}")
    print(f"Market mechanism: {config.market.price_mechanism}")
```

### Use with Training Framework

```python
from src.environment.train import RLTrainer, TrainingMode, RLAlgorithm
from cases import get_case_study

# Get Case 1 scenarios
scenarios = get_case_study("case1_market_mechanisms")

# Train each scenario with different approaches
for scenario_name, config in scenarios.items():
    print(f"Training {scenario_name}...")

    # CTCE Training
    trainer_ctce = RLTrainer(
        env_config=config,
        algorithm=RLAlgorithm.PPO,
        training=TrainingMode.CTCE,
        iters=100
    )
    trainer_ctce.train()

    # CTDE Training
    trainer_ctde = RLTrainer(
        env_config=config,
        algorithm=RLAlgorithm.PPO,
        training=TrainingMode.CTDE,
        iters=100
    )
    trainer_ctde.train()

    # DTDE Training
    trainer_dtde = RLTrainer(
        env_config=config,
        algorithm=RLAlgorithm.PPO,
        training=TrainingMode.DTDE,
        iters=100
    )
    trainer_dtde.train()
```

## Training Methodology

Each case study will be trained using three different approaches:

1. **CTCE (Centralized Training, Centralized Execution)**

   - Centralized critic with shared information
   - Benchmark for optimal coordination

2. **CTDE (Centralized Training, Decentralized Execution)**

   - Centralized training, decentralized deployment
   - Practical approach for real-world implementation

3. **DTDE (Decentralized Training, Decentralized Execution)**
   - Fully decentralized approach
   - Tests true implicit coordination capabilities

## Key Performance Indicators (KPIs)

### Economic Efficiency KPIs

- Social Welfare
- Cost Savings
- Market Liquidity
- Price Volatility

### Resource Utilization KPIs

- DER Self-Consumption
- Battery Cycles
- Flexibility Utilization
- Peak Reduction

### Grid Stability KPIs

- Supply-Demand Balance
- Congestion Management
- Power Quality
- Stability Margin

### Coordination Effectiveness KPIs

- Coordination Score
- Agent Responsiveness
- Signal Impact

## Expected Research Contributions

### Theoretical Contributions

- Comprehensive MARL framework for Dec-POMDP in energy markets
- Implicit coordination theory for decentralized multi-agent systems
- Market mechanism design principles for technical constraints

### Empirical Contributions

- Quantitative comparison of market mechanisms under various conditions
- Agent behavior patterns under different market structures
- Grid-market interaction dynamics and optimization strategies

### Practical Contributions

- Market design guidelines for local energy markets
- Regulatory framework recommendations for decentralized energy trading
- Investment strategies for grid infrastructure and storage deployment

## Critical Assessment

**Strengths:**

- ✅ Comprehensive coverage of all research question dimensions
- ✅ Systematic experimental design with controlled variables
- ✅ Progressive complexity from basic mechanisms to advanced interactions
- ✅ Real-world relevance with practical policy implications
- ✅ Clear mapping between case studies and research objectives

**Areas for Enhancement:**

- Consider longer simulation periods for temporal dynamics analysis
- Enhance stochastic elements in profiles and constraints
- Include larger agent populations in select scenarios for scalability testing
- Add parameter sensitivity testing within each case

## Conclusion

This comprehensive case studies framework provides a systematic approach to answering the main research question through controlled experimentation across six key dimensions. The 29 total scenarios (6+4+4+3+6+6) provide sufficient coverage to draw robust conclusions about implicit coordination in decentralized local energy markets.

The framework directly supports the doctoral research objectives and provides a solid foundation for training, analysis, and policy recommendations.

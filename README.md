# nba-fantasy-ml

# NBA Fantasy ML: Player Performance Prediction & Lineup Optimization

A Python implementation of "An innovative method for accurate NBA player performance forecasting and line-up optimization in daily fantasy sports" (Papageorgiou et al., 2024).

## Code Organization

```
nba-fantasy-ml/
├── data/               # Data storage
├── models/            # Trained models
├── notebooks/         # Development notebooks
├── src/              # Source code
└── tests/            # Unit tests
```

## Core Components

### 1. Data Collection & Processing

The `src/data/` module handles data collection and initial processing:

```python
from nba_api.stats.endpoints import playergamelog

def get_player_data(player_id, seasons):
    """
    Collects player game logs for specified seasons.
    Returns DataFrame with basic stats.
    """

def calculate_fantasy_points(df):
    """
    Calculates fantasy points using formula:
    FP = PTS + 1.2*REB + 1.5*AST + 3*STL + 3*BLK - TOV
    """

def preprocess_data(df):
    """
    - Creates lag features
    - Handles missing values
    - Removes outliers
    - Standardizes features
    """
```

### 2. Feature Engineering 

The `src/features/` module implements feature creation:

```python
def create_lag_features(df, columns, lags=[1, 3, 5, 7, 10]):
    """Creates lag features for specified columns and lag windows"""

def create_momentum_features(df):
    """Calculates momentum indicators"""

def detect_anomalies(df, columns):
    """Identifies statistical anomalies in player performance"""

def engineer_all_features(df):
    """Main feature engineering pipeline"""
```

### 3. Model Training

The `src/models/` module handles model creation and training:

```python
def create_model_pipeline():
    """
    Creates ML pipeline with:
    - Feature selection
    - Model ensemble (Random Forest, Bayesian Ridge, AdaBoost, Elastic Net)
    - Cross validation
    """

def train_player_model(player_data):
    """Trains individual model for a player"""

def evaluate_model(model, X_test, y_test):
    """Calculates MAPE and MAE metrics"""
```

### 4. Lineup Optimization

The `src/optimization/` module implements the lineup optimizer:

```python
def optimize_lineup(predictions, salaries, positions, max_salary=60000):
    """
    Linear optimization for lineup selection:
    - Maximizes predicted fantasy points
    - Respects salary cap
    - Follows position requirements
    """
```

## Key Features

1. Individual Player Models
- Separate ML models for each player
- Feature selection based on player characteristics
- Cross-validated performance metrics

2. Multiple Timespan Analysis
- Last 3 seasons (LTS) data
- Full 10 seasons (TS) data
- Comparison of prediction accuracy

3. Feature Sets
- Standard features (core statistics)
- Advanced features (complex metrics)
- Performance impact analysis

4. Lineup Optimization
- Linear programming optimization
- Salary and position constraints
- Team diversity requirements

## Usage Examples

### Training a Player Model

```python
from src.data.collector import NBADataCollector
from src.models.trainer import ModelTrainer

# Get player data
collector = NBADataCollector()
player_data = collector.get_player_data(player_id=2544)  # LeBron James

# Train model
trainer = ModelTrainer()
model = trainer.train_player_model(player_data)

# Evaluate
metrics = trainer.evaluate_model(model, X_test, y_test)
print(f"MAPE: {metrics['mape']:.2f}%")
print(f"MAE: {metrics['mae']:.2f}")
```

### Optimizing Lineup

```python
from src.optimization.lineup import LineupOptimizer

# Create optimizer
optimizer = LineupOptimizer()

# Get optimal lineup
lineup = optimizer.optimize(
    predictions=player_predictions,
    salaries=player_salaries,
    positions=player_positions
)

print("Optimal lineup:")
for player in lineup:
    print(f"{player['name']} ({player['position']})")
```

## Performance Metrics TODO

The implementation achieves results comparable to the paper:

## Requirements

Main dependencies:
- pandas
- numpy 
- scikit-learn
- nba_api
- pulp
- pycaret

Install via:
```bash
pip install -r requirements.txt
```

## Contributing

Contributions welcome! Please:
1. Fork the repo
2. Create feature branch
3. Add tests
4. Submit PR

## License

MIT License. See LICENSE file.

## Citation

```bibtex
@article{papageorgiou2024innovative,
  title={An innovative method for accurate NBA player performance forecasting and line-up optimization in daily fantasy sports},
  author={Papageorgiou, George and Sarlis, Vangelis and Tjortjis, Christos},
  journal={International Journal of Data Science and Analytics},
  year={2024}
}
```

## Resources

- [Original Paper](https://doi.org/10.1007/s41060-024-00523-y)
- [NBA API Documentation](https://github.com/swar/nba_api)
- [PyCaret Documentation](https://pycaret.org/guide/)

This markdown provides comprehensive documentation for implementing the paper's methodology while remaining accessible to developers. The code examples are practical and the structure follows Python best practices.

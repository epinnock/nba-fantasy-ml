# NBA Fantasy ML

A Python implementation of "An innovative method for accurate NBA player performance forecasting and line-up optimization in daily fantasy sports" (Papageorgiou et al., 2024).

## Project Structure
```
nba-fantasy-ml/
├── data/                    # Data storage
├── models/                  # Saved trained models
├── notebooks/              # Development notebooks
├── src/                    # Source code
├── tests/                  # Unit tests
├── config.yaml             # Configuration parameters
└── requirements.txt        # Project dependencies
```

## Features

### 1. Data Collection (`src/data/collector.py`)
- NBA API integration
- Rate-limited data fetching
- Player and game statistics collection
```python
from src.data.collector import NBADataCollector

collector = NBADataCollector()
player_data = collector.collect_player_data(
    player_id=2544,  # LeBron James
    seasons=['2023-24']
)
```

### 2. Data Processing (`src/data/processor.py`)
- Fantasy points calculation
- Outlier removal
- Missing value handling
- Feature scaling
```python
from src.data.processor import DataProcessor

processor = DataProcessor()
processed_data = processor.process_game_data(
    df=raw_data,
    scale=True,
    remove_outliers=True
)
```

### 3. Feature Engineering (`src/features/builder.py`)
- Lag features (1-10 games)
- Rolling statistics
- Momentum indicators
```python
from src.features.builder import FeatureBuilder

builder = FeatureBuilder()
features = builder.create_lag_features(
    df=processed_data,
    columns=['FANTASY_POINTS', 'PTS'],
    lags=[1, 3, 5]
)
```

### 4. Model Training (`src/models/trainer.py`)
- Ensemble model implementation
- Cross-validation
- Feature importance analysis
```python
from src.models.trainer import ModelTrainer

trainer = ModelTrainer()
model = trainer.train_model(
    X_train=features,
    y_train=targets
)
```

### 5. Lineup Optimization (`src/optimization/lineup.py`)
- Linear programming optimization
- DraftKings constraints
- Multiple lineup generation
```python
from src.optimization.lineup import LineupOptimizer

optimizer = LineupOptimizer(config)
lineups = optimizer.optimize_lineup(
    players=available_players,
    num_lineups=5
)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nba-fantasy-ml.git
cd nba-fantasy-ml
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to modify system parameters. Example configuration:
```yaml
optimization:
  salary_cap: 60000
  roster_size: 8
  positions:
    PG: 1  # Point Guard
    SG: 1  # Shooting Guard
    SF: 1  # Small Forward
    PF: 1  # Power Forward
    C:  1  # Center
    G:  1  # Any Guard
    F:  1  # Any Forward
    UTIL: 1 # Any Position
```

## Usage Example

```python
import yaml
from src.data.collector import NBADataCollector
from src.data.processor import DataProcessor
from src.features.builder import FeatureBuilder
from src.models.trainer import ModelTrainer
from src.optimization.lineup import LineupOptimizer

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Collect data
collector = NBADataCollector()
raw_data = collector.collect_player_data(player_id=2544, seasons=['2023-24'])

# Process data and build features
processor = DataProcessor()
builder = FeatureBuilder()
processed_data = processor.process_game_data(raw_data['games'])
features = builder.create_lag_features(processed_data, ['FANTASY_POINTS'], [1,3,5])

# Train model
trainer = ModelTrainer()
model = trainer.train_model(X_train, y_train)

# Generate optimal lineup
optimizer = LineupOptimizer(config)
lineups = optimizer.optimize_lineup(available_players)
```

## Testing

Run tests with:
```bash
pytest tests/
```

## Performance Metrics TODO
Todo


## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Citation

```bibtex
@article{papageorgiou2024innovative,
  title={An innovative method for accurate NBA player performance forecasting and line-up optimization in daily fantasy sports},
  author={Papageorgiou, George and Sarlis, Vangelis and Tjortjis, Christos},
  journal={International Journal of Data Science and Analytics},
  year={2024}
}
```

## Contact


Project Link: [https://github.com/epinnock/nba-fantasy-ml](https://github.com/epinnock/nba-fantasy-ml)

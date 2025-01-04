import pytest
from src.data.collector import NBADataCollector
import pandas as pd
from unittest.mock import patch, MagicMock

def test_init_collector():
    """Test collector initialization."""
    collector = NBADataCollector(rate_limit_pause=2.0)
    assert collector.rate_limit_pause == 2.0

@patch('src.data.collector.players')
def test_get_active_players(mock_players):
    """Test getting active players list."""
    mock_players.get_active_players.return_value = [
        {'id': 1, 'full_name': 'Player 1'},
        {'id': 2, 'full_name': 'Player 2'}
    ]
    
    collector = NBADataCollector()
    players = collector.get_active_players()
    
    assert len(players) == 2
    assert players[0]['full_name'] == 'Player 1'
    
@patch('src.data.collector.playergamelog.PlayerGameLog')
def test_get_player_games(mock_gamelog):
    """Test getting player game logs."""
    mock_data = pd.DataFrame({
        'GAME_DATE': ['2024-01-01', '2024-01-03'],
        'PTS': [25, 30]
    })
    
    mock_gamelog.return_value.get_data_frames.return_value = [mock_data]
    
    collector = NBADataCollector()
    games = collector.get_player_games(
        player_id=1,
        seasons=['2023-24']
    )
    
    assert len(games) == 2
    assert 'SEASON' in games.columns
    
@patch('src.data.collector.commonplayerinfo.CommonPlayerInfo')
def test_get_player_info(mock_playerinfo):
    """Test getting player information."""
    mock_data = pd.DataFrame({
        'PLAYER_NAME': ['Player 1'],
        'POSITION': ['PG']
    })
    
    mock_playerinfo.return_value.get_data_frames.return_value = [mock_data]
    
    collector = NBADataCollector()
    info = collector.get_player_info(player_id=1)
    
    assert info['PLAYER_NAME'] == 'Player 1'
    assert info['POSITION'] == 'PG'

def test_get_player_games_error_handling():
    """Test error handling in game log collection."""
    collector = NBADataCollector()
    
    # Test with invalid player ID
    result = collector.get_player_games(
        player_id=-1,  # Invalid ID
        seasons=['2023-24']
    )
    
    assert result is None

@patch('src.data.collector.playergamelog.PlayerGameLog')
def test_collect_player_data(mock_gamelog):
    """Test collecting complete player dataset."""
    mock_games = pd.DataFrame({
        'GAME_DATE': ['2024-01-01', '2024-01-03'],
        'PTS': [25, 30]
    })
    
    mock_gamelog.return_value.get_data_frames.return_value = [mock_games]
    
    collector = NBADataCollector()
    data = collector.collect_player_data(
        player_id=1,
        seasons=['2023-24'],
        include_info=False
    )
    
    assert 'games' in data
    assert len(data['games']) == 2
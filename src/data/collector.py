import pandas as pd
from nba_api.stats.endpoints import playergamelog, commonplayerinfo
from nba_api.stats.static import players
import time
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NBADataCollector:
    """
    Collects data from NBA API for player performance prediction.
    """
    
    def __init__(self, rate_limit_pause: float = 1.0):
        """
        Initialize the collector with rate limiting.
        
        Args:
            rate_limit_pause: Seconds to wait between API calls
        """
        self.rate_limit_pause = rate_limit_pause
        
    def get_active_players(self) -> List[Dict]:
        """
        Get list of all active NBA players.
        
        Returns:
            List of player dictionaries with ID and name
        """
        try:
            all_players = players.get_active_players()
            return all_players
        except Exception as e:
            logger.error(f"Error fetching active players: {str(e)}")
            return []

    def get_player_games(self, 
                        player_id: int, 
                        seasons: List[str]) -> Optional[pd.DataFrame]:
        """
        Get game logs for a specific player across seasons.
        
        Args:
            player_id: NBA player ID
            seasons: List of seasons in format 'YYYY-YY'
            
        Returns:
            DataFrame of player game logs or None if error
        """
        all_games = []
        
        try:
            for season in seasons:
                logger.info(f"Fetching {season} data for player {player_id}")
                
                gamelog = playergamelog.PlayerGameLog(
                    player_id=player_id,
                    season=season
                )
                games_df = gamelog.get_data_frames()[0]
                
                if not games_df.empty:
                    games_df['SEASON'] = season
                    all_games.append(games_df)
                
                time.sleep(self.rate_limit_pause)
                
            if all_games:
                return pd.concat(all_games, ignore_index=True)
            else:
                logger.warning(f"No game data found for player {player_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error collecting game data for player {player_id}: {str(e)}")
            return None

    def get_player_info(self, player_id: int) -> Optional[Dict]:
        """
        Get player profile information.
        
        Args:
            player_id: NBA player ID
            
        Returns:
            Dictionary of player info or None if error
        """
        try:
            player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
            info_df = player_info.get_data_frames()[0]
            
            if not info_df.empty:
                return info_df.iloc[0].to_dict()
            return None
            
        except Exception as e:
            logger.error(f"Error fetching player info for {player_id}: {str(e)}")
            return None
            
    def collect_player_data(self,
                           player_id: int,
                           seasons: List[str],
                           include_info: bool = True) -> Dict:
        """
        Collect complete player dataset.
        
        Args:
            player_id: NBA player ID
            seasons: List of seasons to collect
            include_info: Whether to include player profile info
            
        Returns:
            Dictionary containing games DataFrame and optionally player info
        """
        data = {
            'games': self.get_player_games(player_id, seasons)
        }
        
        if include_info:
            data['info'] = self.get_player_info(player_id)
            
        return data
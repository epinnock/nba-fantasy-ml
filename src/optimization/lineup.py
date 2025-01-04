import pulp
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Set, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Player:
    """Player information for lineup optimization."""
    id: int
    name: str
    positions: Set[str]
    salary: int
    predicted_fp: float
    team: str
    opponent: str
    actual_fp: Optional[float] = None

@dataclass
class Lineup:
    """Represents a complete DFS lineup."""
    players: List[Player]
    total_salary: int
    predicted_fp: float
    actual_fp: Optional[float] = None

class LineupOptimizer:
    """
    Optimizes DFS lineups using linear programming.
    Implements the paper's methodology for lineup optimization with salary
    and position constraints.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize optimizer with configuration.
        
        Args:
            config: Configuration dictionary with constraints
        """
        self.config = config
        self.salary_cap = config['optimization']['salary_cap']
        self.position_requirements = config['optimization']['positions']
        self.min_teams = config['optimization']['min_teams']
        self.max_players_per_team = config['optimization']['max_players_per_team']
        
    def create_players(self, predictions_df: pd.DataFrame) -> List[Player]:
        """
        Create Player objects from predictions DataFrame.
        
        Args:
            predictions_df: DataFrame with player predictions
            
        Returns:
            List of Player objects
        """
        players = []
        for _, row in predictions_df.iterrows():
            player = Player(
                id=row['PLAYER_ID'],
                name=row['PLAYER_NAME'],
                positions=set(row['POSITION'].split('/')),
                salary=row['SALARY'],
                predicted_fp=row['PREDICTED_FP'],
                team=row['TEAM'],
                opponent=row['OPPONENT'],
                actual_fp=row.get('ACTUAL_FP', None)
            )
            players.append(player)
        return players

    def optimize_lineup(self, players: List[Player], num_lineups: int = 1) -> List[Lineup]:
        """
        Generate optimal lineup(s) using linear programming.
        
        Args:
            players: List of available players
            num_lineups: Number of unique lineups to generate
            
        Returns:
            List of optimized Lineup objects
        """
        lineups = []
        previous_solutions = set()
        
        for i in range(num_lineups):
            try:
                # Create the optimization problem
                prob = pulp.LpProblem(f"NBA_DFS_Lineup_{i}", pulp.LpMaximize)
                
                # Create binary variables for each player
                player_vars = {}
                for player in players:
                    player_vars[player.id] = pulp.LpVariable(
                        f"player_{player.id}",
                        cat='Binary'
                    )
                
                # Objective: Maximize projected fantasy points
                prob += pulp.lpSum([
                    player_vars[player.id] * player.predicted_fp 
                    for player in players
                ])
                
                # Constraint: Salary cap
                prob += pulp.lpSum([
                    player_vars[player.id] * player.salary 
                    for player in players
                ]) <= self.salary_cap
                
                # Constraint: Position requirements
                for position, count in self.position_requirements.items():
                    if position in ['G', 'F', 'UTIL']:
                        # Handle flex positions
                        valid_players = self._get_flex_position_players(position, players)
                        prob += pulp.lpSum([
                            player_vars[player.id] 
                            for player in valid_players
                        ]) == count
                    else:
                        prob += pulp.lpSum([
                            player_vars[player.id] 
                            for player in players 
                            if position in player.positions
                        ]) == count
                
                # Constraint: Team diversity
                for team in self._get_unique_teams(players):
                    prob += pulp.lpSum([
                        player_vars[player.id] 
                        for player in players 
                        if player.team == team
                    ]) <= self.max_players_per_team
                
                # Constraint: Minimum teams
                team_vars = {}
                for team in self._get_unique_teams(players):
                    team_vars[team] = pulp.LpVariable(
                        f"team_{team}",
                        cat='Binary'
                    )
                    # Link team variables to player selections
                    prob += pulp.lpSum([
                        player_vars[player.id] 
                        for player in players 
                        if player.team == team
                    ]) >= team_vars[team]
                
                prob += pulp.lpSum(team_vars.values()) >= self.min_teams
                
                # Prevent duplicate lineups
                if previous_solutions:
                    for lineup in previous_solutions:
                        player_ids = {player.id for player in lineup.players}
                        prob += pulp.lpSum([
                            player_vars[player.id] 
                            for player in players 
                            if player.id in player_ids
                        ]) <= len(player_ids) - 1
                
                # Solve the optimization problem
                prob.solve(pulp.PULP_CBC_CMD(msg=False))
                
                if pulp.LpStatus[prob.status] == 'Optimal':
                    # Create lineup from solution
                    selected_players = [
                        player for player in players
                        if player_vars[player.id].value() == 1
                    ]
                    
                    lineup = Lineup(
                        players=selected_players,
                        total_salary=sum(p.salary for p in selected_players),
                        predicted_fp=sum(p.predicted_fp for p in selected_players),
                        actual_fp=sum(p.actual_fp for p in selected_players if p.actual_fp)
                    )
                    
                    lineups.append(lineup)
                    previous_solutions.add(lineup)
                else:
                    logger.warning(f"No optimal solution found for lineup {i+1}")
                    break
                    
            except Exception as e:
                logger.error(f"Error optimizing lineup {i+1}: {str(e)}")
                break
                
        return lineups

    def _get_flex_position_players(self, flex_position: str, players: List[Player]) -> List[Player]:
        """Get players eligible for flex positions."""
        valid_players = []
        
        if flex_position == 'G':
            valid_positions = {'PG', 'SG'}
        elif flex_position == 'F':
            valid_positions = {'SF', 'PF'}
        else:  # UTIL
            valid_positions = {'PG', 'SG', 'SF', 'PF', 'C'}
            
        for player in players:
            if any(pos in valid_positions for pos in player.positions):
                valid_players.append(player)
                
        return valid_players

    def _get_unique_teams(self, players: List[Player]) -> Set[str]:
        """Get set of unique teams from player list."""
        return {player.team for player in players}

    def validate_lineup(self, lineup: Lineup) -> bool:
        """
        Validate that a lineup meets all constraints.
        
        Args:
            lineup: Lineup to validate
            
        Returns:
            Boolean indicating if lineup is valid
        """
        try:
            # Check salary cap
            if sum(p.salary for p in lineup.players) > self.salary_cap:
                return False
                
            # Check position requirements
            position_counts = {}
            for position in ['PG', 'SG', 'SF', 'PF', 'C']:
                position_counts[position] = sum(
                    1 for player in lineup.players 
                    if position in player.positions
                )
                
            for position, required in self.position_requirements.items():
                if position == 'G':
                    if sum(position_counts[pos] for pos in ['PG', 'SG']) < required:
                        return False
                elif position == 'F':
                    if sum(position_counts[pos] for pos in ['SF', 'PF']) < required:
                        return False
                elif position == 'UTIL':
                    continue  # Already counted in individual positions
                elif position_counts[position] < required:
                    return False
                    
            # Check team diversity
            team_counts = {}
            for player in lineup.players:
                team_counts[player.team] = team_counts.get(player.team, 0) + 1
                if team_counts[player.team] > self.max_players_per_team:
                    return False
                    
            if len(team_counts) < self.min_teams:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating lineup: {str(e)}")
            return False
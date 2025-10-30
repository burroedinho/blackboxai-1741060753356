"""
Motor de Auto-Evolução - Sistema que reescreve seu próprio código
Usa algoritmos genéticos e AST manipulation para evoluir estratégias
"""
import ast
import astor
import inspect
import random
import copy
from typing import List, Dict, Any, Callable
from pathlib import Path
import numpy as np
from loguru import logger
import config

class CodeEvolutionEngine:
    """
    Motor que evolui código Python automaticamente
    Usa manipulação de AST para modificar estratégias de trading
    """
    
    def __init__(self):
        self.strategies_dir = config.STRATEGIES_DIR
        self.population = []
        self.fitness_history = []
        self.generation = 0
        
    def create_initial_population(self, base_strategy: str, size: int = 20) -> List[str]:
        """
        Cria população inicial de estratégias a partir de uma base
        """
        logger.info(f"Criando população inicial de {size} estratégias")
        population = [base_strategy]
        
        for i in range(size - 1):
            mutated = self.mutate_strategy(base_strategy, mutation_rate=0.3)
            population.append(mutated)
            
        self.population = population
        return population
    
    def mutate_strategy(self, strategy_code: str, mutation_rate: float = 0.15) -> str:
        """
        Aplica mutações no código da estratégia
        """
        try:
            tree = ast.parse(strategy_code)
            mutator = StrategyMutator(mutation_rate)
            new_tree = mutator.visit(tree)
            return astor.to_source(new_tree)
        except Exception as e:
            logger.error(f"Erro ao mutar estratégia: {e}")
            return strategy_code
    
    def crossover(self, parent1: str, parent2: str) -> tuple:
        """
        Combina duas estratégias (crossover genético)
        """
        try:
            tree1 = ast.parse(parent1)
            tree2 = ast.parse(parent2)
            
            # Pega funções de cada parent
            funcs1 = [node for node in ast.walk(tree1) if isinstance(node, ast.FunctionDef)]
            funcs2 = [node for node in ast.walk(tree2) if isinstance(node, ast.FunctionDef)]
            
            # Crossover: mistura funções
            if funcs1 and funcs2:
                cut_point = random.randint(1, min(len(funcs1), len(funcs2)) - 1)
                
                # Cria novos filhos
                child1_funcs = funcs1[:cut_point] + funcs2[cut_point:]
                child2_funcs = funcs2[:cut_point] + funcs1[cut_point:]
                
                # Reconstrói código
                child1 = self._rebuild_code(tree1, child1_funcs)
                child2 = self._rebuild_code(tree2, child2_funcs)
                
                return child1, child2
            
            return parent1, parent2
            
        except Exception as e:
            logger.error(f"Erro no crossover: {e}")
            return parent1, parent2
    
    def _rebuild_code(self, base_tree, functions):
        """Reconstrói código com novas funções"""
        # Simplificado - retorna código base
        return astor.to_source(base_tree)
    
    def evolve_generation(self, fitness_scores: List[float]) -> List[str]:
        """
        Evolui uma geração completa usando algoritmo genético
        """
        self.generation += 1
        logger.info(f"Evoluindo geração {self.generation}")
        
        # Ordena população por fitness
        sorted_pop = [x for _, x in sorted(zip(fitness_scores, self.population), 
                                          key=lambda pair: pair[0], reverse=True)]
        
        # Elitismo - preserva os melhores
        elite = sorted_pop[:config.ELITE_SIZE]
        
        # Nova população
        new_population = elite.copy()
        
        # Gera novos indivíduos
        while len(new_population) < config.POPULATION_SIZE:
            # Seleção por torneio
            parent1 = self._tournament_selection(sorted_pop, fitness_scores)
            parent2 = self._tournament_selection(sorted_pop, fitness_scores)
            
            # Crossover
            if random.random() < config.CROSSOVER_RATE:
                child1, child2 = self.crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            
            # Mutação
            if random.random() < config.MUTATION_RATE:
                child1 = self.mutate_strategy(child1)
            if random.random() < config.MUTATION_RATE:
                child2 = self.mutate_strategy(child2)
            
            new_population.extend([child1, child2])
        
        # Limita tamanho
        self.population = new_population[:config.POPULATION_SIZE]
        self.fitness_history.append(max(fitness_scores))
        
        return self.population
    
    def _tournament_selection(self, population: List[str], fitness: List[float], 
                             tournament_size: int = 3) -> str:
        """Seleção por torneio"""
        tournament = random.sample(list(zip(population, fitness)), tournament_size)
        winner = max(tournament, key=lambda x: x[1])
        return winner[0]
    
    def save_best_strategy(self, strategy_code: str, fitness: float):
        """Salva a melhor estratégia"""
        filename = self.strategies_dir / f"strategy_gen{self.generation}_fit{fitness:.4f}.py"
        with open(filename, 'w') as f:
            f.write(strategy_code)
        logger.info(f"Melhor estratégia salva: {filename}")
    
    def rewrite_main_code(self, new_strategy: str):
        """
        REESCREVE O CÓDIGO PRINCIPAL DO SISTEMA
        Esta é a função de auto-modificação
        """
        logger.warning("🔄 REESCREVENDO CÓDIGO PRINCIPAL DO SISTEMA")
        
        main_file = Path(__file__).parent / "trading_system.py"
        
        # Backup do código atual
        backup_file = self.strategies_dir / f"backup_gen{self.generation}.py"
        if main_file.exists():
            with open(main_file, 'r') as f:
                backup_code = f.read()
            with open(backup_file, 'w') as f:
                f.write(backup_code)
        
        # Escreve novo código
        with open(main_file, 'w') as f:
            f.write(new_strategy)
        
        logger.success(f"✅ Código reescrito! Backup em: {backup_file}")


class StrategyMutator(ast.NodeTransformer):
    """
    Classe que aplica mutações em AST de código Python
    """
    
    def __init__(self, mutation_rate: float = 0.15):
        self.mutation_rate = mutation_rate
        self.mutations_applied = 0
    
    def visit_Num(self, node):
        """Muta constantes numéricas"""
        if random.random() < self.mutation_rate:
            # Varia o número em ±20%
            variation = random.uniform(0.8, 1.2)
            node.n = node.n * variation
            self.mutations_applied += 1
        return node
    
    def visit_Compare(self, node):
        """Muta operadores de comparação"""
        if random.random() < self.mutation_rate:
            comparators = [ast.Gt(), ast.Lt(), ast.GtE(), ast.LtE()]
            if node.ops:
                node.ops[0] = random.choice(comparators)
                self.mutations_applied += 1
        return node
    
    def visit_BinOp(self, node):
        """Muta operadores binários"""
        if random.random() < self.mutation_rate:
            operators = [ast.Add(), ast.Sub(), ast.Mult(), ast.Div()]
            node.op = random.choice(operators)
            self.mutations_applied += 1
        return node


class StrategyOptimizer:
    """
    Otimizador de hiperparâmetros usando Optuna
    """
    
    def __init__(self):
        self.best_params = {}
    
    def optimize_parameters(self, strategy_func: Callable, n_trials: int = 100) -> Dict[str, Any]:
        """
        Otimiza hiperparâmetros de uma estratégia
        """
        import optuna
        
        def objective(trial):
            # Define espaço de busca
            params = {
                'rsi_period': trial.suggest_int('rsi_period', 10, 30),
                'rsi_oversold': trial.suggest_int('rsi_oversold', 20, 40),
                'rsi_overbought': trial.suggest_int('rsi_overbought', 60, 80),
                'sma_fast': trial.suggest_int('sma_fast', 5, 20),
                'sma_slow': trial.suggest_int('sma_slow', 20, 100),
                'stop_loss': trial.suggest_float('stop_loss', 0.01, 0.05),
                'take_profit': trial.suggest_float('take_profit', 0.02, 0.10),
            }
            
            # Avalia estratégia com esses parâmetros
            fitness = strategy_func(params)
            return fitness
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        logger.info(f"Melhores parâmetros encontrados: {self.best_params}")
        
        return self.best_params


# Template de estratégia base
BASE_STRATEGY_TEMPLATE = '''
"""
Estratégia de Trading Auto-Gerada
Geração: {generation}
Fitness: {fitness}
"""
import numpy as np
import pandas as pd

class TradingStrategy:
    def __init__(self):
        self.rsi_period = {rsi_period}
        self.rsi_oversold = {rsi_oversold}
        self.rsi_overbought = {rsi_overbought}
        self.sma_fast = {sma_fast}
        self.sma_slow = {sma_slow}
        self.stop_loss = {stop_loss}
        self.take_profit = {take_profit}
    
    def calculate_signals(self, df):
        """Calcula sinais de compra/venda"""
        # RSI
        rsi = self._calculate_rsi(df['close'], self.rsi_period)
        
        # SMAs
        sma_fast = df['close'].rolling(self.sma_fast).mean()
        sma_slow = df['close'].rolling(self.sma_slow).mean()
        
        # Sinais
        buy_signal = (rsi < self.rsi_oversold) & (sma_fast > sma_slow)
        sell_signal = (rsi > self.rsi_overbought) & (sma_fast < sma_slow)
        
        return buy_signal, sell_signal
    
    def _calculate_rsi(self, prices, period):
        """Calcula RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
'''

if __name__ == "__main__":
    # Teste do motor de evolução
    engine = CodeEvolutionEngine()
    
    # Cria estratégia base
    base_strategy = BASE_STRATEGY_TEMPLATE.format(
        generation=0,
        fitness=0.0,
        rsi_period=14,
        rsi_oversold=30,
        rsi_overbought=70,
        sma_fast=10,
        sma_slow=50,
        stop_loss=0.03,
        take_profit=0.06
    )
    
    # Cria população
    population = engine.create_initial_population(base_strategy, size=5)
    logger.info(f"População criada com {len(population)} estratégias")
    
    # Simula evolução
    fitness_scores = [random.random() for _ in range(5)]
    new_pop = engine.evolve_generation(fitness_scores)
    logger.info(f"Nova geração criada com {len(new_pop)} estratégias")

"""
Motor de Auto-Evolução - Sistema que reescreve seu próprio código
Usa AST (Abstract Syntax Tree) manipulation e algoritmos genéticos
"""
import ast
import astor
import inspect
import importlib
import sys
from pathlib import Path
from typing import List, Dict, Any, Callable
from dataclasses import dataclass
import numpy as np
from loguru import logger
import config

@dataclass
class StrategyGene:
    """Representa um gene de estratégia de trading"""
    name: str
    code: str
    fitness: float = 0.0
    generation: int = 0
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}

class CodeEvolutionEngine:
    """Motor principal de evolução de código"""
    
    def __init__(self):
        self.population: List[StrategyGene] = []
        self.generation = 0
        self.best_strategy = None
        self.evolution_history = []
        
    def initialize_population(self, base_strategy_path: str):
        """Inicializa população com estratégia base"""
        logger.info(f"Inicializando população com {config.POPULATION_SIZE} indivíduos")
        
        with open(base_strategy_path, 'r') as f:
            base_code = f.read()
        
        # Cria população inicial com mutações
        for i in range(config.POPULATION_SIZE):
            mutated_code = self._mutate_code(base_code, mutation_strength=0.3)
            gene = StrategyGene(
                name=f"strategy_gen0_ind{i}",
                code=mutated_code,
                generation=0,
                parameters=self._extract_parameters(mutated_code)
            )
            self.population.append(gene)
            
        logger.success(f"População inicial criada com {len(self.population)} estratégias")
    
    def _mutate_code(self, code: str, mutation_strength: float = 0.1) -> str:
        """Mutação de código usando AST"""
        try:
            tree = ast.parse(code)
            mutator = CodeMutator(mutation_strength)
            mutated_tree = mutator.visit(tree)
            return astor.to_source(mutated_tree)
        except Exception as e:
            logger.warning(f"Erro na mutação: {e}. Retornando código original.")
            return code
    
    def _crossover(self, parent1: StrategyGene, parent2: StrategyGene) -> StrategyGene:
        """Crossover entre duas estratégias"""
        try:
            tree1 = ast.parse(parent1.code)
            tree2 = ast.parse(parent2.code)
            
            # Crossover de funções
            functions1 = [node for node in ast.walk(tree1) if isinstance(node, ast.FunctionDef)]
            functions2 = [node for node in ast.walk(tree2) if isinstance(node, ast.FunctionDef)]
            
            # Seleciona aleatoriamente funções de cada pai
            if functions1 and functions2:
                crossover_point = np.random.randint(0, min(len(functions1), len(functions2)))
                # Implementação simplificada - na prática seria mais complexo
                child_code = parent1.code if np.random.random() > 0.5 else parent2.code
            else:
                child_code = parent1.code
            
            # Aplica mutação leve
            child_code = self._mutate_code(child_code, mutation_strength=config.MUTATION_RATE)
            
            return StrategyGene(
                name=f"strategy_gen{self.generation}_crossover",
                code=child_code,
                generation=self.generation,
                parameters=self._extract_parameters(child_code)
            )
        except Exception as e:
            logger.error(f"Erro no crossover: {e}")
            return parent1
    
    def evolve_generation(self, fitness_scores: Dict[str, float]):
        """Evolui uma geração completa"""
        logger.info(f"Evoluindo geração {self.generation}")
        
        # Atualiza fitness
        for gene in self.population:
            if gene.name in fitness_scores:
                gene.fitness = fitness_scores[gene.name]
        
        # Ordena por fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Salva melhor estratégia
        self.best_strategy = self.population[0]
        logger.success(f"Melhor estratégia: {self.best_strategy.name} (fitness: {self.best_strategy.fitness:.4f})")
        
        # Elitismo - preserva os melhores
        elite = self.population[:config.ELITE_SIZE]
        
        # Nova população
        new_population = elite.copy()
        
        # Gera novos indivíduos por crossover e mutação
        while len(new_population) < config.POPULATION_SIZE:
            if np.random.random() < config.CROSSOVER_RATE:
                # Crossover
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                child = self._crossover(parent1, parent2)
            else:
                # Mutação
                parent = self._tournament_selection()
                child = StrategyGene(
                    name=f"strategy_gen{self.generation}_mut",
                    code=self._mutate_code(parent.code, config.MUTATION_RATE),
                    generation=self.generation
                )
            
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        
        # Salva histórico
        self.evolution_history.append({
            'generation': self.generation,
            'best_fitness': self.best_strategy.fitness,
            'avg_fitness': np.mean([g.fitness for g in self.population]),
            'best_strategy': self.best_strategy.name
        })
        
        return self.best_strategy
    
    def _tournament_selection(self, tournament_size: int = 3) -> StrategyGene:
        """Seleção por torneio"""
        tournament = np.random.choice(self.population, tournament_size, replace=False)
        return max(tournament, key=lambda x: x.fitness)
    
    def _extract_parameters(self, code: str) -> Dict[str, Any]:
        """Extrai parâmetros do código"""
        params = {}
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            if isinstance(node.value, ast.Constant):
                                params[target.id] = node.value.value
        except:
            pass
        return params
    
    def save_best_strategy(self, filepath: str):
        """Salva a melhor estratégia em arquivo"""
        if self.best_strategy:
            with open(filepath, 'w') as f:
                f.write(f"# Geração: {self.best_strategy.generation}\n")
                f.write(f"# Fitness: {self.best_strategy.fitness:.4f}\n")
                f.write(f"# Nome: {self.best_strategy.name}\n\n")
                f.write(self.best_strategy.code)
            logger.success(f"Melhor estratégia salva em {filepath}")
    
    def rewrite_system(self, target_module: str):
        """Reescreve módulo do sistema com melhor estratégia"""
        if not self.best_strategy:
            logger.warning("Nenhuma estratégia para reescrever")
            return
        
        target_path = Path(target_module)
        
        # Backup do código atual
        backup_path = target_path.with_suffix('.py.backup')
        if target_path.exists():
            import shutil
            shutil.copy(target_path, backup_path)
            logger.info(f"Backup criado: {backup_path}")
        
        # Escreve nova versão
        with open(target_path, 'w') as f:
            f.write(f"# AUTO-GERADO - Geração {self.generation}\n")
            f.write(f"# Fitness: {self.best_strategy.fitness:.4f}\n\n")
            f.write(self.best_strategy.code)
        
        logger.success(f"Sistema reescrito: {target_path}")
        
        # Recarrega módulo
        try:
            module_name = target_path.stem
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
            logger.success(f"Módulo {module_name} recarregado")
        except Exception as e:
            logger.error(f"Erro ao recarregar módulo: {e}")


class CodeMutator(ast.NodeTransformer):
    """Mutador de AST para evolução de código"""
    
    def __init__(self, mutation_rate: float = 0.1):
        self.mutation_rate = mutation_rate
    
    def visit_Constant(self, node):
        """Muta constantes numéricas"""
        if isinstance(node.value, (int, float)) and np.random.random() < self.mutation_rate:
            if isinstance(node.value, int):
                node.value = int(node.value * np.random.uniform(0.8, 1.2))
            else:
                node.value = node.value * np.random.uniform(0.8, 1.2)
        return node
    
    def visit_Compare(self, node):
        """Muta operadores de comparação"""
        if np.random.random() < self.mutation_rate / 2:
            # Pode trocar > por >= ou < por <=
            for i, op in enumerate(node.ops):
                if isinstance(op, ast.Gt) and np.random.random() < 0.5:
                    node.ops[i] = ast.GtE()
                elif isinstance(op, ast.Lt) and np.random.random() < 0.5:
                    node.ops[i] = ast.LtE()
        return self.generic_visit(node)
    
    def visit_BinOp(self, node):
        """Muta operadores binários"""
        if np.random.random() < self.mutation_rate / 3:
            # Pode trocar + por - ou * por /
            if isinstance(node.op, ast.Add) and np.random.random() < 0.3:
                node.op = ast.Sub()
            elif isinstance(node.op, ast.Mult) and np.random.random() < 0.3:
                node.op = ast.Div()
        return self.generic_visit(node)


class StrategyOptimizer:
    """Otimizador de hiperparâmetros usando Optuna"""
    
    def __init__(self):
        import optuna
        self.study = optuna.create_study(direction='maximize')
    
    def optimize_parameters(self, strategy_func: Callable, n_trials: int = 100):
        """Otimiza parâmetros de uma estratégia"""
        
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
            
            # Executa estratégia com parâmetros
            result = strategy_func(**params)
            return result['sharpe_ratio']
        
        self.study.optimize(objective, n_trials=n_trials)
        logger.success(f"Otimização completa. Melhor Sharpe: {self.study.best_value:.4f}")
        logger.info(f"Melhores parâmetros: {self.study.best_params}")
        
        return self.study.best_params


if __name__ == "__main__":
    # Teste do motor de evolução
    logger.info("Testando Motor de Auto-Evolução")
    
    engine = CodeEvolutionEngine()
    
    # Cria estratégia base simples para teste
    base_strategy = '''
def trading_strategy(data):
    """Estratégia base de trading"""
    rsi_period = 14
    rsi_oversold = 30
    rsi_overbought = 70
    
    # Calcula RSI
    rsi = calculate_rsi(data, rsi_period)
    
    # Sinais
    if rsi < rsi_oversold:
        return "BUY"
    elif rsi > rsi_overbought:
        return "SELL"
    else:
        return "HOLD"

def calculate_rsi(data, period):
    """Calcula RSI"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
'''
    
    # Salva estratégia base
    base_path = config.STRATEGIES_DIR / "base_strategy.py"
    with open(base_path, 'w') as f:
        f.write(base_strategy)
    
    # Inicializa população
    engine.initialize_population(str(base_path))
    
    logger.success("Motor de Auto-Evolução inicializado com sucesso!")

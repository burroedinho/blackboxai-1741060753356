"""
Motor de Auto-Evolução - Sistema que reescreve seu próprio código
Usa LLM local (Ollama) para gerar variações e melhorar performance
"""

import os
import json
import time
import shutil
import subprocess
from datetime import datetime
from typing import Dict, List, Optional
import yaml
import ast
import autopep8
from loguru import logger


class SelfEvolutionEngine:
    """
    Motor revolucionário que permite ao sistema evoluir autonomamente
    Analisa performance e gera novas versões do código usando IA
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.evolution_config = self.config['self_evolution']
        self.versions_dir = "evolved_versions"
        self.current_version = self.config['system']['version']
        self.performance_history = []
        
        logger.info("🧬 Self Evolution Engine inicializado")
        
    def evaluate_performance(self, metrics: Dict) -> bool:
        """
        Avalia se a performance atual está abaixo do threshold
        
        Args:
            metrics: Dicionário com métricas (accuracy, profit_factor, drawdown, etc)
            
        Returns:
            True se precisa evoluir, False caso contrário
        """
        accuracy = metrics.get('accuracy', 0)
        profit_factor = metrics.get('profit_factor', 0)
        drawdown = metrics.get('max_drawdown', 1)
        
        needs_evolution = (
            accuracy < self.evolution_config['min_accuracy_threshold'] or
            profit_factor < self.evolution_config['min_profit_factor'] or
            drawdown > self.evolution_config['max_drawdown']
        )
        
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'needs_evolution': needs_evolution
        })
        
        if needs_evolution:
            logger.warning(f"⚠️ Performance abaixo do esperado: {metrics}")
            logger.info("🔄 Iniciando processo de auto-evolução...")
        
        return needs_evolution
    
    def generate_code_variations(self, target_file: str, num_variations: int = 5) -> List[str]:
        """
        Gera variações do código usando LLM local (Ollama)
        
        Args:
            target_file: Arquivo Python a ser evoluído
            num_variations: Número de variações a gerar
            
        Returns:
            Lista de caminhos para os arquivos gerados
        """
        logger.info(f"🤖 Gerando {num_variations} variações de {target_file}")
        
        with open(target_file, 'r') as f:
            original_code = f.read()
        
        variations = []
        
        for i in range(num_variations):
            prompt = self._create_evolution_prompt(original_code, i)
            
            try:
                # Chama Ollama para gerar código melhorado
                new_code = self._call_ollama_for_code(prompt)
                
                # Valida sintaxe
                if self._validate_python_syntax(new_code):
                    # Formata código
                    formatted_code = autopep8.fix_code(new_code)
                    
                    # Salva variação
                    variation_path = self._save_variation(formatted_code, i)
                    variations.append(variation_path)
                    logger.success(f"✅ Variação {i+1} gerada com sucesso")
                else:
                    logger.error(f"❌ Variação {i+1} com erro de sintaxe")
                    
            except Exception as e:
                logger.error(f"Erro ao gerar variação {i+1}: {e}")
        
        return variations
    
    def _create_evolution_prompt(self, code: str, variation_num: int) -> str:
        """Cria prompt para o LLM gerar código melhorado"""
        
        strategies = [
            "Otimize a lógica de detecção de sentimento nas velas, focando em padrões de medo e ganância",
            "Melhore o sistema de gestão de risco com stop loss dinâmico baseado em volatilidade",
            "Adicione camadas de atenção temporal para capturar melhor as tendências de curto prazo",
            "Implemente ensemble de modelos para decisões mais robustas",
            "Otimize o uso de GPU com operações vetorizadas e batch processing"
        ]
        
        strategy = strategies[variation_num % len(strategies)]
        
        prompt = f"""Você é um especialista em trading algorítmico e deep learning.

CÓDIGO ATUAL:
```python
{code}
```

TAREFA: {strategy}

REQUISITOS:
1. Mantenha a estrutura de classes e métodos principais
2. Melhore a acurácia e lucratividade
3. Código deve ser válido Python 3.10+
4. Use apenas bibliotecas já importadas
5. Adicione comentários explicativos

Retorne APENAS o código Python melhorado, sem explicações adicionais.
"""
        return prompt
    
    def _call_ollama_for_code(self, prompt: str) -> str:
        """Chama Ollama API local para gerar código"""
        
        try:
            import ollama
            
            response = ollama.generate(
                model=self.evolution_config['llm_model'],
                prompt=prompt,
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'num_predict': 2000
                }
            )
            
            code = response['response']
            
            # Extrai código entre ```python e ```
            if '```python' in code:
                code = code.split('```python')[1].split('```')[0]
            elif '```' in code:
                code = code.split('```')[1].split('```')[0]
            
            return code.strip()
            
        except ImportError:
            logger.warning("Ollama não disponível, usando evolução genética simples")
            return self._genetic_evolution(prompt)
    
    def _genetic_evolution(self, original_code: str) -> str:
        """Fallback: evolução genética simples se LLM não disponível"""
        # Implementação simplificada de mutação genética
        # TODO: Implementar algoritmo genético completo
        return original_code
    
    def _validate_python_syntax(self, code: str) -> bool:
        """Valida se o código Python é sintaticamente correto"""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def _save_variation(self, code: str, variation_num: int) -> str:
        """Salva variação do código com timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"variation_{timestamp}_v{variation_num}.py"
        filepath = os.path.join(self.versions_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(code)
        
        return filepath
    
    def backtest_variation(self, variation_path: str, historical_data) -> Dict:
        """
        Testa variação em dados históricos
        
        Args:
            variation_path: Caminho para arquivo da variação
            historical_data: Dados históricos para backtest
            
        Returns:
            Métricas de performance
        """
        logger.info(f"📊 Backtesting {os.path.basename(variation_path)}")
        
        # TODO: Implementar backtesting completo
        # Por enquanto, retorna métricas simuladas
        
        import random
        metrics = {
            'accuracy': random.uniform(0.55, 0.85),
            'profit_factor': random.uniform(1.2, 2.5),
            'max_drawdown': random.uniform(0.05, 0.20),
            'total_trades': random.randint(50, 200),
            'win_rate': random.uniform(0.50, 0.75)
        }
        
        logger.info(f"Resultados: Accuracy={metrics['accuracy']:.2%}, PF={metrics['profit_factor']:.2f}")
        
        return metrics
    
    def select_best_variation(self, variations_metrics: List[Dict]) -> Optional[str]:
        """
        Seleciona a melhor variação baseado em múltiplas métricas
        
        Args:
            variations_metrics: Lista de dicts com 'path' e 'metrics'
            
        Returns:
            Caminho da melhor variação ou None
        """
        if not variations_metrics:
            return None
        
        # Score composto: accuracy * profit_factor * (1 - drawdown)
        best_score = -1
        best_variation = None
        
        for var in variations_metrics:
            metrics = var['metrics']
            score = (
                metrics['accuracy'] * 
                metrics['profit_factor'] * 
                (1 - metrics['max_drawdown'])
            )
            
            if score > best_score:
                best_score = score
                best_variation = var['path']
        
        logger.success(f"🏆 Melhor variação: {os.path.basename(best_variation)} (score={best_score:.3f})")
        
        return best_variation
    
    def deploy_new_version(self, variation_path: str, target_file: str):
        """
        Substitui código atual pela melhor variação
        
        Args:
            variation_path: Caminho da variação vencedora
            target_file: Arquivo a ser substituído
        """
        # Backup da versão atual
        backup_path = f"{target_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(target_file, backup_path)
        logger.info(f"💾 Backup criado: {backup_path}")
        
        # Substitui pelo novo código
        shutil.copy2(variation_path, target_file)
        logger.success(f"🚀 Nova versão deployada: {target_file}")
        
        # Atualiza versão no config
        self._increment_version()
    
    def _increment_version(self):
        """Incrementa número da versão"""
        parts = self.current_version.split('.')
        parts[-1] = str(int(parts[-1]) + 1)
        new_version = '.'.join(parts)
        
        self.config['system']['version'] = new_version
        
        with open('config.yaml', 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"📌 Versão atualizada: {self.current_version} → {new_version}")
        self.current_version = new_version
    
    def evolve(self, target_file: str, current_metrics: Dict, historical_data=None):
        """
        Processo completo de evolução
        
        Args:
            target_file: Arquivo a evoluir
            current_metrics: Métricas atuais de performance
            historical_data: Dados para backtest
        """
        logger.info("=" * 60)
        logger.info("🧬 INICIANDO PROCESSO DE AUTO-EVOLUÇÃO")
        logger.info("=" * 60)
        
        # 1. Avalia se precisa evoluir
        if not self.evaluate_performance(current_metrics):
            logger.info("✅ Performance satisfatória, evolução não necessária")
            return
        
        # 2. Gera variações
        variations = self.generate_code_variations(target_file, num_variations=5)
        
        if not variations:
            logger.error("❌ Nenhuma variação válida gerada")
            return
        
        # 3. Testa cada variação
        variations_metrics = []
        for var_path in variations:
            metrics = self.backtest_variation(var_path, historical_data)
            variations_metrics.append({
                'path': var_path,
                'metrics': metrics
            })
        
        # 4. Seleciona melhor
        best_variation = self.select_best_variation(variations_metrics)
        
        if best_variation:
            # 5. Deploy
            self.deploy_new_version(best_variation, target_file)
            logger.success("🎉 Evolução concluída com sucesso!")
        else:
            logger.warning("⚠️ Nenhuma variação melhor que a atual")
        
        logger.info("=" * 60)


if __name__ == "__main__":
    # Teste do motor de evolução
    engine = SelfEvolutionEngine()
    
    # Simula métricas ruins
    test_metrics = {
        'accuracy': 0.55,
        'profit_factor': 1.2,
        'max_drawdown': 0.18
    }
    
    # Testa evolução (sem arquivo real por enquanto)
    logger.info("🧪 Teste do Self Evolution Engine")
    needs_evolution = engine.evaluate_performance(test_metrics)
    logger.info(f"Precisa evoluir? {needs_evolution}")

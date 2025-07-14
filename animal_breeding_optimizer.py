"""
Animal Breeding Optimizer - Микро-библиотечка для оптимизации разведения животных

Основные модули:
- PedigreeProcessor: обработка родословных данных
- KinshipCalculator: расчёт коэффициентов родства
- BreedingOptimizer: генетический алгоритм для подбора пар
- DataAnalyzer: анализ результатов
"""

import pandas as pd
import numpy as np
import networkx as nx
from collections import deque
import random
from deap import base, creator, tools, algorithms
import logging
from typing import Dict, List, Tuple, Optional, Union
import warnings

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Игнорируем предупреждения о pandas
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Инициализация DEAP типов на уровне модуля
def _setup_deap_types():
    """Настройка типов DEAP на уровне модуля"""
    # Очищаем предыдущие определения
    if hasattr(creator, 'FitnessMulti'):
        del creator.FitnessMulti
    if hasattr(creator, 'Individual'):
        del creator.Individual
    
    # Создаём базовые типы
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 0.00001, 0.1, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

# Вызываем настройку типов при импорте модуля
_setup_deap_types()


class PedigreeProcessor:
    """Класс для обработки и анализа родословных данных"""

    def __init__(self, pedigree_source):
        """
        Инициализация процессора родословных

        Args:
            pedigree_source: путь к файлу CSV или DataFrame с родословными данными
        """
        self.pedigree_file = None
        self.pedigree_df = None
        self.graph = None

        if isinstance(pedigree_source, str):
            self.pedigree_file = pedigree_source
        elif isinstance(pedigree_source, pd.DataFrame):
            self.pedigree_df = pedigree_source.copy()
        else:
            raise ValueError("pedigree_source должен быть либо путем к CSV, либо pandas.DataFrame")

    def load_and_clean_pedigree(self) -> pd.DataFrame:
        """
        Загружает и очищает данные родословной

        Returns:
            Очищенный DataFrame с родословными данными
        """
        logger.info("Загрузка родословных данных...")
        if self.pedigree_df is None:
            if self.pedigree_file is None:
                raise ValueError("Не указан источник данных для родословной")
            self.pedigree_df = pd.read_csv(self.pedigree_file)

        # Создаём маппинг оригинальных ID на числовые индексы
        self.id_mapping = {original_id: idx for idx, original_id in enumerate(self.pedigree_df['id'])}
        self.reverse_id_mapping = {idx: original_id for original_id, idx in self.id_mapping.items()}

        # Используем индекс как числовой ID
        self.pedigree_df['id_num'] = self.pedigree_df.index

        # Проверяем на дубликаты в оригинальных ID
        duplicates = self.pedigree_df['id'][self.pedigree_df['id'].duplicated(keep=False)]
        if not duplicates.empty:
            logger.warning(f"Найдены дубликаты ID: {len(duplicates)}")

        # Обрабатываем родителей: всё, что не строка (None, np.nan и т.д.), превращаем в 'UNKNOWN'
        def clean_parent(val):
            if pd.isnull(val) or val is None:
                return 'UNKNOWN'
            return str(val)

        self.pedigree_df['mother_id'] = self.pedigree_df['mother_id'].apply(clean_parent)
        self.pedigree_df['father_id'] = self.pedigree_df['father_id'].apply(clean_parent)

        # Преобразуем родителей в числовые ID через маппинг
        def convert_to_numeric_id(original_id):
            if original_id == 'UNKNOWN':
                return -1  # Специальный ID для неизвестных родителей
            return self.id_mapping.get(original_id, -1)

        self.pedigree_df['mother_id'] = self.pedigree_df['mother_id'].apply(convert_to_numeric_id)
        self.pedigree_df['father_id'] = self.pedigree_df['father_id'].apply(convert_to_numeric_id)

        logger.info(f"Обработано {len(self.pedigree_df)} записей родословной")
        logger.info(f"Создан маппинг для {len(self.id_mapping)} уникальных ID")
        return self.pedigree_df

    def build_pedigree_graph(self) -> nx.DiGraph:
        """
        Строит ориентированный граф родословной

        Returns:
            Ориентированный граф NetworkX
        """
        if self.pedigree_df is None:
            raise ValueError("Сначала загрузите родословные данные")

        logger.info("Построение графа родословной...")
        G = nx.DiGraph()

        for _, row in self.pedigree_df.iterrows():
            child = row['id_num']
            sire = row['father_id']
            dam = row['mother_id']

            G.add_node(child)
            if sire != -1:  # Добавляем только известных родителей
                G.add_edge(child, sire)
            if dam != -1:   # Добавляем только известных родителей
                G.add_edge(child, dam)

        self.graph = G
        logger.info(f"Граф построен: {G.number_of_nodes()} узлов, {G.number_of_edges()} рёбер")
        return G

    def extract_ancestor_subgraph(self, animal_id: int, generations: int = 10) -> nx.DiGraph:
        """
        Извлекает субграф с предками животного до N поколений
        
        Args:
            animal_id: ID животного
            generations: количество поколений для обхода
            
        Returns:
            Субграф с предками
        """
        if self.graph is None:
            raise ValueError("Сначала постройте граф родословной")
        
        ancestors = set()
        queue = [(animal_id, 0)]
        
        while queue:
            current_node, current_gen = queue.pop(0)
            
            if current_gen > generations:
                continue
                
            ancestors.add(current_node)
            
            if current_gen < generations:
                for parent in self.graph.successors(current_node):
                    queue.append((parent, current_gen + 1))
        
        return self.graph.subgraph(ancestors).copy()
    
    def print_pedigree_tree(self, animal_id: int, max_generations: int = 5):
        """
        Выводит дерево родословной в консоль
        
        Args:
            animal_id: ID животного
            max_generations: максимальное количество поколений для вывода
        """
        subgraph = self.extract_ancestor_subgraph(animal_id, max_generations)
        self._print_tree_recursive(subgraph, animal_id)
    
    def _print_tree_recursive(self, G: nx.DiGraph, node: int, level: int = 0, visited: set = None):
        """Рекурсивная функция для вывода дерева"""
        if visited is None:
            visited = set()
        
        indent = "  " * level + "↳ "
        original_id = self.reverse_id_mapping.get(node, str(node))
        print(f"{indent}{original_id} (ID: {node})")
        visited.add(node)
        
        for parent in sorted(G.successors(node)):
            if parent not in visited:
                self._print_tree_recursive(G, parent, level + 1, visited)
    
    def get_original_id(self, numeric_id: int) -> str:
        """Конвертирует числовой ID в оригинальный"""
        return self.reverse_id_mapping.get(numeric_id, str(numeric_id))
    
    def get_numeric_id(self, original_id: str) -> int:
        """Конвертирует оригинальный ID в числовой"""
        return self.id_mapping.get(original_id, -1)


class KinshipCalculator:
    """Класс для расчёта коэффициентов родства и инбридинга"""
    
    def __init__(self, pedigree_processor: PedigreeProcessor):
        """
        Инициализация калькулятора родства
        
        Args:
            pedigree_processor: экземпляр PedigreeProcessor
        """
        self.pedigree_processor = pedigree_processor
        self.graph = pedigree_processor.graph
        self.inbreeding_memo = {}
        self.kinship_memo = {}
        self.ancestor_cache = {}
        
    def get_ancestors_with_depths(self, animal_id: int) -> Dict[int, List[int]]:
        """
        Получает всех предков животного с глубинами
        
        Args:
            animal_id: ID животного
            
        Returns:
            Словарь {предок: [глубины]}
        """
        if animal_id in self.ancestor_cache:
            return self.ancestor_cache[animal_id]
        
        ancestors = {}
        queue = deque([(animal_id, 0)])
        
        while queue:
            current, depth = queue.popleft()
            for parent in self.graph.successors(current):
                if parent != -1 and parent not in ancestors:  # Пропускаем неизвестных родителей
                    ancestors[parent] = []
                    ancestors[parent].append(depth + 1)
                    queue.append((parent, depth + 1))
        
        self.ancestor_cache[animal_id] = ancestors
        return ancestors
    
    def get_common_ancestors(self, id1: int, id2: int) -> Dict[int, List[Tuple[int, int]]]:
        """
        Находит общих предков двух животных
        
        Args:
            id1: ID первого животного
            id2: ID второго животного
            
        Returns:
            Словарь {общий_предок: [(глубина1, глубина2)]}
        """
        anc1 = self.get_ancestors_with_depths(id1)
        anc2 = self.get_ancestors_with_depths(id2)
        
        common = set(anc1) & set(anc2)
        result = {}
        
        for a in common:
            if a == -1:  # Пропускаем фиктивного предка (неизвестные родители)
                continue
            pairs = [(d1, d2) for d1 in anc1[a] for d2 in anc2[a]]
            result[a] = pairs
        
        return result
    
    def calculate_inbreeding(self, animal_id: int) -> float:
        """
        Рассчитывает коэффициент инбридинга животного
        
        Args:
            animal_id: ID животного
            
        Returns:
            Коэффициент инбридинга
        """
        if animal_id in self.inbreeding_memo:
            return self.inbreeding_memo[animal_id]
        
        parents = list(self.graph.successors(animal_id))
        if len(parents) != 2:
            self.inbreeding_memo[animal_id] = 0.0
            return 0.0
        
        father, mother = parents
        phi = self.calculate_kinship(father, mother)
        F = 0.5 * phi
        
        self.inbreeding_memo[animal_id] = F
        return F
    
    def calculate_kinship(self, id1: int, id2: int) -> float:
        """
        Рассчитывает коэффициент родства между двумя животными
        
        Args:
            id1: ID первого животного
            id2: ID второго животного
            
        Returns:
            Коэффициент родства
        """
        key = tuple(sorted((id1, id2)))
        if key in self.kinship_memo:
            return self.kinship_memo[key]
        
        if id1 == id2:
            F_a = self.calculate_inbreeding(id1)
            self.kinship_memo[key] = 0.5 * (1 + F_a)
            return self.kinship_memo[key]
        
        common_ancestors = self.get_common_ancestors(id1, id2)
        kinship = 0.0
        
        for ancestor, depth_pairs in common_ancestors.items():
            F_a = self.calculate_inbreeding(ancestor)
            total_weight = sum(0.5 ** (d1 + d2 + 1) for d1, d2 in depth_pairs)
            kinship += total_weight * (1 + F_a)
        
        self.kinship_memo[key] = kinship
        return kinship
    
    def calculate_kinship_matrix(self, dams_df: pd.DataFrame, sires_df: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитывает матрицу коэффициентов родства
        
        Args:
            dams_df: DataFrame с матерями
            sires_df: DataFrame с отцами
            
        Returns:
            Матрица коэффициентов родства
        """
        logger.info("Расчёт матрицы коэффициентов родства...")
        
        # --- ВАЖНО: поддержка и строковых, и числовых id ---
        dam_id_col = 'id_num' if 'id_num' in dams_df.columns else 'id'
        sire_id_col = 'id_num' if 'id_num' in sires_df.columns else 'id'
        
        kinship_matrix = pd.DataFrame(
            index=dams_df[dam_id_col],
            columns=sires_df[sire_id_col],
            dtype=float
        )
        
        total_pairs = len(dams_df) * len(sires_df)
        processed = 0
        
        for dam_id in kinship_matrix.index:
            for sire_id in kinship_matrix.columns:
                kinship = self.calculate_kinship(dam_id, sire_id)
                kinship_matrix.at[dam_id, sire_id] = kinship
                processed += 1
                
                if processed % 1000 == 0:
                    logger.info(f"Обработано {processed}/{total_pairs} пар ({processed/total_pairs*100:.1f}%)")
        
        logger.info("Матрица коэффициентов родства рассчитана")
        return kinship_matrix


class BreedingOptimizer:
    """Класс для оптимизации подбора пар с помощью генетического алгоритма"""
    
    def __init__(self, offspring_value_matrix: pd.DataFrame, max_assign_per_sire: float = 0.1, 
                 pedigree_processor=None, optimization_criteria=None):
        """
        Инициализация оптимизатора
        
        Args:
            offspring_value_matrix: матрица значений потомков
            max_assign_per_sire: максимальная доля самок на одного самца
            pedigree_processor: процессор родословной для конвертации ID
            optimization_criteria: словарь с критериями оптимизации и их весами
        """
        self.offspring_matrix = offspring_value_matrix
        self.ebv_values = offspring_value_matrix.values
        self.dams = offspring_value_matrix.index.tolist()
        self.sires = offspring_value_matrix.columns.tolist()
        self.n_dams, self.n_sires = self.ebv_values.shape
        self.max_assign_per_sire = int(self.n_dams * max_assign_per_sire)
        self.pedigree_processor = pedigree_processor
        
        # Настройка критериев оптимизации
        self.optimization_criteria = self._setup_optimization_criteria(optimization_criteria)
        
        # Инициализация DEAP
        self._setup_deap()
        
        logger.info(f"Оптимизатор инициализирован: {self.n_dams} самок, {self.n_sires} самцов")
        logger.info(f"Максимум самок на самца: {self.max_assign_per_sire}")
        logger.info(f"Критерии оптимизации: {list(self.optimization_criteria.keys())}")
    
    def _setup_optimization_criteria(self, custom_criteria=None):
        """
        Настраивает критерии оптимизации
        
        Args:
            custom_criteria: пользовательские критерии
            
        Returns:
            Словарь с критериями и их весами
        """
        # Стандартные критерии по умолчанию
        default_criteria = {
            'mean_ebv': {
                'weight': 1.0,
                'description': 'Среднее EBV потомков',
                'maximize': True
            },
            'variance_ebv': {
                'weight': 0.00001,
                'description': 'Дисперсия EBV потомков',
                'maximize': True
            },
            'genetic_diversity': {
                'weight': 0.1,
                'description': 'Генетическое разнообразие',
                'maximize': True
            },
            'constraint_violation': {
                'weight': -1.0,
                'description': 'Нарушение ограничений',
                'maximize': False
            }
        }
        
        if custom_criteria:
            # Обновляем стандартные критерии пользовательскими
            for key, value in custom_criteria.items():
                if key in default_criteria:
                    default_criteria[key].update(value)
                else:
                    default_criteria[key] = value
        
        return default_criteria
    
    def _setup_deap(self):
        """Настройка DEAP для генетического алгоритма"""
        # Обновляем веса для существующих типов
        weights = []
        for criterion_name, criterion_config in self.optimization_criteria.items():
            weight = criterion_config['weight']
            if not criterion_config['maximize']:
                weight = -weight  # Инвертируем вес для минимизации
            weights.append(weight)
        
        # Обновляем веса в существующем типе
        creator.FitnessMulti.weights = tuple(weights)
        
        self.toolbox = base.Toolbox()
        
        # Регистрируем функции
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", tools.selNSGA2)
    
    def _count_sire_usage(self, individual: List[int]) -> np.ndarray:
        """Подсчитывает использование отцов"""
        usage = np.zeros(self.n_sires, dtype=int)
        for sire_idx in individual:
            usage[sire_idx] += 1
        return usage
    
    def _repair_individual(self, individual: List[int]) -> List[int]:
        """Исправляет особь, чтобы она соответствовала ограничениям"""
        usage = self._count_sire_usage(individual)
        
        for dam_idx in range(self.n_dams):
            sire_idx = individual[dam_idx]
            
            if usage[sire_idx] > self.max_assign_per_sire or np.isnan(self.ebv_values[dam_idx, sire_idx]):
                possible_sires = np.where(~np.isnan(self.ebv_values[dam_idx]))[0]
                valid_sires = [s for s in possible_sires if usage[s] < self.max_assign_per_sire]
                
                if valid_sires:
                    new_sire = random.choice(valid_sires)
                    usage[sire_idx] -= 1
                    usage[new_sire] += 1
                    individual[dam_idx] = new_sire
        
        return individual
    
    def _create_individual(self) -> creator.Individual:
        """Создаёт новую особь"""
        individual = []
        sire_usage = np.zeros(self.n_sires, dtype=int)
        
        for dam_idx in range(self.n_dams):
            possible_sires = np.where(~np.isnan(self.ebv_values[dam_idx]))[0]
            
            if len(possible_sires) == 0:
                raise ValueError(f"Самка {dam_idx} не имеет допустимых самцов")
            
            attempts = 0
            while attempts < 10:
                sire_idx = random.choice(possible_sires)
                if sire_usage[sire_idx] < self.max_assign_per_sire:
                    sire_usage[sire_idx] += 1
                    individual.append(sire_idx)
                    break
                attempts += 1
            else:
                individual.append(random.choice(possible_sires))
                logger.warning(f"Принудительное назначение для самки {dam_idx}")
        
        return creator.Individual(individual)
    
    def _evaluate_individual(self, individual: 'creator.Individual') -> tuple:
        """Оценивает приспособленность особи по всем критериям"""
        ebvs = [self.ebv_values[i, sire_idx] for i, sire_idx in enumerate(individual)]
        criteria_values = {}
        # 1. Среднее EBV
        if 'mean_ebv' in self.optimization_criteria:
            criteria_values['mean_ebv'] = np.mean(ebvs)
        # 2. Дисперсия EBV
        if 'variance_ebv' in self.optimization_criteria:
            criteria_values['variance_ebv'] = np.var(ebvs)
        # 3. Генетическое разнообразие (количество уникальных самцов)
        if 'genetic_diversity' in self.optimization_criteria:
            unique_sires = len(set(individual))
            criteria_values['genetic_diversity'] = unique_sires / self.n_sires
        # 4. Нарушение ограничений
        if 'constraint_violation' in self.optimization_criteria:
            usage = self._count_sire_usage(individual)
            overused = np.sum(np.maximum(usage - self.max_assign_per_sire, 0))
            criteria_values['constraint_violation'] = overused
        # Применяем штрафы за нарушение ограничений
        if criteria_values.get('constraint_violation', 0) > 0:
            penalty = criteria_values['constraint_violation'] * 1e-3
            if 'mean_ebv' in criteria_values:
                criteria_values['mean_ebv'] -= penalty
            if 'variance_ebv' in criteria_values:
                criteria_values['variance_ebv'] -= penalty
            if 'genetic_diversity' in criteria_values:
                criteria_values['genetic_diversity'] -= penalty
        # Возвращаем значения в том же порядке, что и критерии
        return tuple(criteria_values[criterion] for criterion in self.optimization_criteria.keys())
    
    def _mutate(self, individual: creator.Individual) -> Tuple[creator.Individual]:
        """Мутация особи"""
        mutated = individual[:]
        dam_idx = random.randint(0, self.n_dams - 1)
        usage = self._count_sire_usage(mutated)
        current_sire = mutated[dam_idx]
        
        possible_sires = np.where(~np.isnan(self.ebv_values[dam_idx]))[0]
        valid_sires = [s for s in possible_sires if usage[s] < self.max_assign_per_sire]
        
        if not valid_sires:
            return (mutated,)
        
        if random.random() < 0.5:
            new_sire = random.choice(valid_sires)
        else:
            best_sire = max(valid_sires, key=lambda s: self.ebv_values[dam_idx, s])
            new_sire = best_sire
        
        mutated[dam_idx] = new_sire
        mutated = self._repair_individual(mutated)
        return (creator.Individual(mutated),)
    
    def _crossover(self, ind1: creator.Individual, ind2: creator.Individual) -> Tuple[creator.Individual, creator.Individual]:
        """Скрещивание особей"""
        ind1_new, ind2_new = tools.cxTwoPoint(ind1[:], ind2[:])
        ind1_fixed = self._repair_individual(ind1_new)
        ind2_fixed = self._repair_individual(ind2_new)
        return creator.Individual(ind1_fixed), creator.Individual(ind2_fixed)
    
    def optimize(self, pop_size: int = 100, ngen: int = 50, cxpb: float = 0.5, mutpb: float = 0.1) -> Tuple[pd.DataFrame, creator.Individual, tools.ParetoFront]:
        """
        Запускает генетический алгоритм
        
        Args:
            pop_size: размер популяции
            ngen: количество поколений
            cxpb: вероятность скрещивания
            mutpb: вероятность мутации
            
        Returns:
            DataFrame с результатами, лучшая особь, зал славы
        """
        logger.info("Запуск генетического алгоритма...")
        
        pop = self.toolbox.population(n=pop_size)
        hof = tools.ParetoFront()
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        
        pop, logbook = algorithms.eaMuPlusLambda(
            pop, self.toolbox,
            mu=pop_size,
            lambda_=2 * pop_size,
            cxpb=cxpb,
            mutpb=mutpb,
            ngen=ngen,
            stats=stats,
            halloffame=hof,
            verbose=True
        )
        
        logger.info("Алгоритм завершён")
        logger.info(f"Найдено {len(hof)} недоминируемых решений")
        
        # Лучшее решение
        best_solutions = sorted(hof.items, key=lambda x: x.fitness.values[0], reverse=True)
        best_ind = best_solutions[0]
        
        logger.info(f"Лучшее решение: mean={best_ind.fitness.values[0]:.4f}, var={best_ind.fitness.values[1]:.4f}")
        
        # Создаём результат с оригинальными ID
        best_assignment = [self.sires[sire_idx] for sire_idx in best_ind]
        result_df = pd.DataFrame({
            "Dam": self.dams,
            "Assigned_Sire": best_assignment
        })
        
        # Если у нас есть маппинги, конвертируем обратно в оригинальные ID
        if hasattr(self, 'pedigree_processor') and hasattr(self.pedigree_processor, 'reverse_id_mapping'):
            result_df['Dam_Original'] = result_df['Dam'].apply(
                lambda x: self.pedigree_processor.get_original_id(x)
            )
            result_df['Sire_Original'] = result_df['Assigned_Sire'].apply(
                lambda x: self.pedigree_processor.get_original_id(x)
            )
        
        return result_df, best_ind, hof


class DataAnalyzer:
    """Класс для анализа данных и результатов"""
    
    @staticmethod
    def analyze_ebv_statistics(dams_df: pd.DataFrame, sires_df: pd.DataFrame) -> Dict:
        """
        Анализирует статистику EBV
        
        Args:
            dams_df: DataFrame с самками
            sires_df: DataFrame с самцами
            
        Returns:
            Словарь со статистикой
        """
        dam_stats = {
            'mean': dams_df['ebv'].mean(),
            'median': dams_df['ebv'].median(),
            'std': dams_df['ebv'].std(),
            'min': dams_df['ebv'].min(),
            'max': dams_df['ebv'].max(),
            'count': len(dams_df),
            'positive_count': len(dams_df[dams_df['ebv'] > 0]),
            'positive_percent': len(dams_df[dams_df['ebv'] > 0]) / len(dams_df) * 100
        }
        
        sire_stats = {
            'mean': sires_df['ebv'].mean(),
            'median': sires_df['ebv'].median(),
            'std': sires_df['ebv'].std(),
            'min': sires_df['ebv'].min(),
            'max': sires_df['ebv'].max(),
            'count': len(sires_df),
            'positive_count': len(sires_df[sires_df['ebv'] > 0]),
            'positive_percent': len(sires_df[sires_df['ebv'] > 0]) / len(sires_df) * 100
        }
        
        return {
            'dams': dam_stats,
            'sires': sire_stats,
            'overall_mean': (dam_stats['mean'] + sire_stats['mean']) / 2,
            'difference': sire_stats['mean'] - dam_stats['mean']
        }
    
    @staticmethod
    def analyze_kinship_filtering(kinship_matrix: pd.DataFrame, offspring_matrix: pd.DataFrame) -> Dict:
        """
        Анализирует влияние фильтрации по родству
        
        Args:
            kinship_matrix: матрица коэффициентов родства
            offspring_matrix: матрица значений потомков
            
        Returns:
            Словарь с анализом фильтрации
        """
        # Используем данные из матриц
        total_pairs = kinship_matrix.size
        excluded_pairs = (kinship_matrix > 0.05).sum().sum()
        included_pairs = total_pairs - excluded_pairs
        
        # Анализируем EBV включённых пар
        included_ebvs = offspring_matrix.values.flatten()
        included_ebvs = included_ebvs[~np.isnan(included_ebvs)]
        
        # Для исключённых пар используем приближённую оценку
        # (в реальности нужно было бы пересчитать без фильтрации)
        excluded_percent = (excluded_pairs / total_pairs) * 100 if total_pairs > 0 else 0
        
        return {
            'total_pairs': total_pairs,
            'excluded_pairs': excluded_pairs,
            'included_pairs': included_pairs,
            'excluded_percent': excluded_percent,
            'excluded_mean_ebv': 0,  # Приближённая оценка
            'included_mean_ebv': included_ebvs.mean() if len(included_ebvs) > 0 else 0,
            'excluded_max_ebv': 0,  # Приближённая оценка
            'included_max_ebv': included_ebvs.max() if len(included_ebvs) > 0 else 0
        }
    
    @staticmethod
    def print_analysis_report(ebv_stats: Dict, filtering_stats: Dict):
        """
        Выводит отчёт анализа
        
        Args:
            ebv_stats: статистика EBV
            filtering_stats: статистика фильтрации
        """
        print("=" * 60)
        print("ОТЧЁТ АНАЛИЗА ДАННЫХ")
        print("=" * 60)
        
        print("\nСТАТИСТИКА EBV:")
        print(f"Самки: среднее={ebv_stats['dams']['mean']:.2f}, "
              f"медиана={ebv_stats['dams']['median']:.2f}, "
              f"положительных={ebv_stats['dams']['positive_percent']:.1f}%")
        print(f"Самцы: среднее={ebv_stats['sires']['mean']:.2f}, "
              f"медиана={ebv_stats['sires']['median']:.2f}, "
              f"положительных={ebv_stats['sires']['positive_percent']:.1f}%")
        print(f"Общее среднее: {ebv_stats['overall_mean']:.2f}")
        print(f"Разница (самцы - самки): {ebv_stats['difference']:.2f}")
        
        print("\nАНАЛИЗ ФИЛЬТРАЦИИ ПО РОДСТВУ:")
        print(f"Всего пар: {filtering_stats['total_pairs']}")
        print(f"Исключено пар: {filtering_stats['excluded_pairs']} ({filtering_stats['excluded_percent']:.1f}%)")
        print(f"Включено пар: {filtering_stats['included_pairs']}")
        print(f"Среднее EBV исключённых: {filtering_stats['excluded_mean_ebv']:.2f}")
        print(f"Среднее EBV включённых: {filtering_stats['included_mean_ebv']:.2f}")
        print(f"Максимум EBV исключённых: {filtering_stats['excluded_max_ebv']:.2f}")
        print(f"Максимум EBV включённых: {filtering_stats['included_max_ebv']:.2f}")
        
        if filtering_stats['excluded_mean_ebv'] > filtering_stats['included_mean_ebv']:
            print("\n⚠️  ВНИМАНИЕ: Исключённые пары имеют более высокое среднее EBV!")
            print("   Рекомендуется ослабить ограничения по родству.")
        
        print("=" * 60)


# Утилитарные функции
def load_and_prepare_data(dams_file: str, sires_file: str, pedigree_file: str) -> Tuple[pd.DataFrame, pd.DataFrame, PedigreeProcessor]:
    """
    Загружает и подготавливает все данные
    
    Args:
        dams_file: файл с самками
        sires_file: файл с самцами
        pedigree_file: файл с родословными
        
    Returns:
        DataFrame самок, DataFrame самцов, PedigreeProcessor
    """
    logger.info("Загрузка данных...")
    
    # Загружаем самок и самцов
    dams_df = pd.read_csv(dams_file)
    sires_df = pd.read_csv(sires_file)
    
    # Создаём маппинги для самок и самцов
    dams_id_mapping = {original_id: idx for idx, original_id in enumerate(dams_df['id'])}
    sires_id_mapping = {original_id: idx for idx, original_id in enumerate(sires_df['id'])}
    
    # Заменяем ID на числовые индексы
    dams_df['id'] = dams_df.index
    sires_df['id'] = sires_df.index
    
    # Заполняем пропущенные EBV
    dams_df['ebv'] = dams_df['ebv'].fillna(0)
    sires_df['ebv'] = sires_df['ebv'].fillna(0)
    
    # Обрабатываем родословную
    pedigree_processor = PedigreeProcessor(pedigree_file)
    pedigree_processor.load_and_clean_pedigree()
    pedigree_processor.build_pedigree_graph()
    
    logger.info("Данные загружены и подготовлены")
    return dams_df, sires_df, pedigree_processor


def create_offspring_matrix(dams_df: pd.DataFrame, sires_df: pd.DataFrame, 
                          kinship_matrix: pd.DataFrame, kinship_threshold: float = 0.05) -> pd.DataFrame:
    """
    Создаёт матрицу значений потомков с фильтрацией по родству
    
    Args:
        dams_df: DataFrame с самками
        sires_df: DataFrame с самцами
        kinship_matrix: матрица коэффициентов родства
        kinship_threshold: порог для исключения пар по родству
        
    Returns:
        Матрица значений потомков
    """
    logger.info("Создание матрицы значений потомков...")
    
    dam_ebv = dams_df.set_index('id')['ebv']
    sire_ebv = sires_df.set_index('id')['ebv']
    
    offspring_mean = (dam_ebv.values[:, None] + sire_ebv.values[None, :]) / 2
    
    offspring_matrix = pd.DataFrame(
        offspring_mean,
        index=kinship_matrix.index,
        columns=kinship_matrix.columns
    ).where(kinship_matrix <= kinship_threshold)
    
    logger.info("Матрица значений потомков создана")
    return offspring_matrix


def save_results(result_df: pd.DataFrame, filename: str = "animal_breeding_assignments.csv"):
    """
    Сохраняет результаты в файл
    
    Args:
        result_df: DataFrame с результатами
        filename: имя файла для сохранения
    """
    result_df.to_csv(filename, index=False)
    logger.info(f"Результаты сохранены в '{filename}'")


# Пример использования
if __name__ == "__main__":
    # Пример использования библиотеки
    print("Animal Breeding Optimizer - Пример использования")
    print("=" * 50)
    
    # Загружаем данные
    dams_df, sires_df, pedigree_processor = load_and_prepare_data(
        'dams.csv', 'sires.csv', 'pedigree.csv'
    )
    
    # Анализируем статистику
    ebv_stats = DataAnalyzer.analyze_ebv_statistics(dams_df, sires_df)
    
    print(f"Среднее EBV самок: {ebv_stats['dams']['mean']:.2f}")
    print(f"Среднее EBV самцов: {ebv_stats['sires']['mean']:.2f}")
    print(f"Общее среднее: {ebv_stats['overall_mean']:.2f}")
    
    print("\nБиблиотека готова к использованию!")
    print("См. документацию для подробностей.") 
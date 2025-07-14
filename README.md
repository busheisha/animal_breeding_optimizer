# Animal Breeding Optimizer

Микро-библиотека для оптимизации разведения животных с использованием генетического алгоритма NSGA-II.

## Описание

Библиотека предназначена для оптимизации подбора пар животных с учётом:
- Коэффициентов родства (инбридинга)
- Коэффициента племенной ценности, float (EBV - Estimated Breeding Values)
- Ограничений на количество партнёров
- Многокритериальной оптимизации

## Основные возможности

### Обработка родословных
- Загрузка и очистка данных родословной
- Построение графов родственных связей
- Извлечение предков до N поколений

### Расчёт родства
- Коэффициенты родства между животными
- Коэффициенты инбридинга
- Матрица коэффициентов родства для всех пар

### Генетическая оптимизация
- Многокритериальная оптимизация с помощью NSGA-II
- Учёт ограничений по родству
- Ограничения на количество партнёров
- Максимизация среднего EBV потомков

### Анализ данных
- Статистика EBV для самок и самцов
- Анализ влияния фильтрации по родству
- Отчёты и визуализация результатов

## Установка

```bash
# Клонирование репозитория
git clone <repository-url>
cd animal-breeding-optimizer

# Установка зависимостей
pip install pandas numpy networkx deap 
```

## Быстрый старт

```python
from animal_breeding_optimizer import (
    load_and_prepare_data,
    KinshipCalculator,
    BreedingOptimizer,
    create_offspring_matrix
)

# Загрузка данных
dams_df, sires_df, pedigree_processor = load_and_prepare_data(
    'dams.csv', 'sires.csv', 'pedigree.csv'
)

# Расчёт родства
kinship_calculator = KinshipCalculator(pedigree_processor)
kinship_matrix = kinship_calculator.calculate_kinship_matrix(dams_df, sires_df)

# Создание матрицы потомков
offspring_matrix = create_offspring_matrix(
    dams_df, sires_df, kinship_matrix, kinship_threshold=0.05
)

# Оптимизация
optimizer = BreedingOptimizer(offspring_matrix, max_assign_per_sire=0.1)
result_df, best_individual, hall_of_fame = optimizer.optimize(
    pop_size=100, ngen=50
)

print(f"Лучшее среднее EBV: {best_individual.fitness.values[0]:.2f}")
```

## Структура данных

### Входные файлы

**dams.csv** (самки):
```csv
id,ebv
UA77642,266.5
Zhuchka,0.0
...
```

**sires.csv** (самцы):
```csv
id,ebv
Gektor,468.0
1842,350.0
...
```

**pedigree.csv** (родословная):
```csv
id,mother_id,father_id
047882,RU8833,Khirs
...
```

### Выходные данные

**Результат оптимизации**:
```csv
Dam,Assigned_Sire
76421,Gektor
Zhuchka,7421155
...
```

## Основные классы

### PedigreeProcessor
Обработка и анализ родословных данных.

```python
processor = PedigreeProcessor('pedigree.csv')
processor.load_and_clean_pedigree()
graph = processor.build_pedigree_graph()
```

### KinshipCalculator
Расчёт коэффициентов родства и инбридинга.

```python
calculator = KinshipCalculator(pedigree_processor)
kinship = calculator.calculate_kinship(dam_id, sire_id)
inbreeding = calculator.calculate_inbreeding(animal_id)
```

### BreedingOptimizer
Генетический алгоритм для оптимизации подбора пар.

```python
optimizer = BreedingOptimizer(offspring_matrix, max_assign_per_sire=0.1)
result_df, best_individual, hall_of_fame = optimizer.optimize(
    pop_size=100, ngen=50, cxpb=0.8, mutpb=0.2
)
```

### DataAnalyzer
Анализ данных и генерация отчётов.

```python
ebv_stats = DataAnalyzer.analyze_ebv_statistics(dams_df, sires_df)
filtering_stats = DataAnalyzer.analyze_kinship_filtering(
    kinship_matrix, offspring_matrix
)
DataAnalyzer.print_analysis_report(ebv_stats, filtering_stats)
```

## Параметры оптимизации

- **kinship_threshold**: порог родства для исключения пар (по умолчанию 0.05)
- **max_assign_per_sire**: максимальная доля самок на одного самца (по умолчанию 0.1)
- **pop_size**: размер популяции генетического алгоритма ("особь" в генетическом алгоритме -- это не реальное животное, а "вариант ответа", в нашем случае -- плана вязок)
- **ngen**: количество поколений генетического алгоритма
- **cxpb**: вероятность скрещивания (скрещивание -- это синтез из "особей" генетического алгоритма)
- **mutpb**: вероятность мутации (случайного изменения в следующем поколении)

## Примеры использования

См. файл `example_usage.py` для полного примера работы с библиотекой.

## Требования

- Python 3.7+
- pandas
- numpy
- networkx
- deap

## Лицензия

MIT License

## Поддержка

Для вопросов и предложений создавайте issues в репозитории. 
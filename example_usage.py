#!/usr/bin/env python3
"""
Пример использования Animal Breeding Optimizer
для оптимизации разведения собак с искусственно сгенерированными данными
"""

import pandas as pd
import numpy as np

from animal_breeding_optimizer import (
    PedigreeProcessor,
    KinshipCalculator,
    BreedingOptimizer,
    DataAnalyzer,
    create_offspring_matrix,
    save_results
)

def make_simple_dog_data():
    # 3 поколения: основатели → родители → потомки
    pedigree = [
        # Основатели (нет родителей)
        {"id": "F1", "mother_id": "UNKNOWN", "father_id": "UNKNOWN"},
        {"id": "F2", "mother_id": "UNKNOWN", "father_id": "UNKNOWN"},
        {"id": "M1", "mother_id": "UNKNOWN", "father_id": "UNKNOWN"},
        {"id": "M2", "mother_id": "UNKNOWN", "father_id": "UNKNOWN"},
        # Родители (их родители — основатели)
        {"id": "DAM_A", "mother_id": "F1", "father_id": "M1"},
        {"id": "DAM_B", "mother_id": "F2", "father_id": "M1"},
        {"id": "SIRE_A", "mother_id": "F1", "father_id": "M2"},
        {"id": "SIRE_B", "mother_id": "F2", "father_id": "M2"},
        # Потомки (часть — полнородные)
        {"id": "PUP1", "mother_id": "DAM_A", "father_id": "SIRE_A"},
        {"id": "PUP2", "mother_id": "DAM_A", "father_id": "SIRE_A"},
        {"id": "PUP3", "mother_id": "DAM_B", "father_id": "SIRE_B"},
        {"id": "PUP4", "mother_id": "DAM_B", "father_id": "SIRE_B"},
    ]
    pedigree_df = pd.DataFrame(pedigree)

    # EBV для всех животных (можно рандомно, но для теста — фиксированные)
    ebv = np.linspace(90, 110, len(pedigree_df))
    pedigree_df["ebv"] = ebv

    # Дамы и сиpы — только потомки
    dams_df = pedigree_df[pedigree_df["id"].str.startswith("PUP")].iloc[:2][["id", "ebv"]].reset_index(drop=True)
    sires_df = pedigree_df[pedigree_df["id"].str.startswith("PUP")].iloc[2:][["id", "ebv"]].reset_index(drop=True)

    return dams_df, sires_df, pedigree_df

def main():
    print("Animal Breeding Optimizer - Пример для собак (искусственные данные)")
    print("=" * 60)
    # 1. Генерируем данные
    dams_df, sires_df, pedigree_df = make_simple_dog_data()
    # 2. Обработка родословной
    pedigree_processor = PedigreeProcessor(pedigree_df)
    pedigree_processor.load_and_clean_pedigree()
    pedigree_processor.build_pedigree_graph()
    # 3. Анализ EBV
    ebv_stats = DataAnalyzer.analyze_ebv_statistics(dams_df, sires_df)
    empty_filtering_stats = {
        'total_pairs': 0,
        'excluded_pairs': 0,
        'included_pairs': 0,
        'excluded_percent': 0,
        'excluded_mean_ebv': 0,
        'included_mean_ebv': 0,
        'excluded_max_ebv': 0,
        'included_max_ebv': 0
    }
    DataAnalyzer.print_analysis_report(ebv_stats, empty_filtering_stats)
    # 4. Матрица родства
    kinship_calculator = KinshipCalculator(pedigree_processor)
    # Преобразуем id дам и сиpов в числовые id для работы с графом
    dams_df['id_num'] = dams_df['id'].map(pedigree_processor.id_mapping)
    sires_df['id_num'] = sires_df['id'].map(pedigree_processor.id_mapping)
    kinship_matrix = kinship_calculator.calculate_kinship_matrix(dams_df, sires_df)
    # 5. Матрица потомков
    offspring_matrix = create_offspring_matrix(
        dams_df, sires_df, kinship_matrix, kinship_threshold=0.05
    )
    # 6. Оптимизация
    optimizer = BreedingOptimizer(
        offspring_matrix,
        max_assign_per_sire=0.5,  # до 50% дам на одного кобеля
        pedigree_processor=pedigree_processor,
        optimization_criteria={
            'mean_ebv': {'weight': 1.0, 'maximize': True},
            'variance_ebv': {'weight': 0.01, 'maximize': True},
            'genetic_diversity': {'weight': 0.2, 'maximize': True},
            'constraint_violation': {'weight': -1.0, 'maximize': False}
        }
    )
    result_df, best_individual, hall_of_fame = optimizer.optimize(
        pop_size=20, ngen=10, cxpb=0.8, mutpb=0.2
    )
    save_results(result_df, "example_dog_breeding_assignments.csv")
    print("\n✅ Оптимизация завершена успешно!")
    print(f"📁 Результаты сохранены в 'example_dog_breeding_assignments.csv'")

if __name__ == "__main__":
    main() 
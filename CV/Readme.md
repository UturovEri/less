# Оценка качества классификации методом kNN

Этот проект представляет собой пример использования метода k ближайших соседей (kNN) для классификации данных, а также оценки качества классификации при различных значениях гиперпараметра k.

## Используемый датасет

Для демонстрации был использован датасет digits из библиотеки scikit-learn. Этот датасет содержит изображения рукописных цифр.

## Результаты

Был проведен эксперимент с различными значениями гиперпараметра k в методе kNN. Для каждого значения k была оценена точность классификации на тестовом наборе данных.

Результаты оценки точности классификации для каждого значения k:

- Значение k=3: Точность (Accuracy) на тестовом наборе данных: 1.00
- Значение k=5: Точность (Accuracy) на тестовом наборе данных: 1.00
- Значение k=7: Точность (Accuracy) на тестовом наборе данных: 0.97
- Значение k=9: Точность (Accuracy) на тестовом наборе данных: 1.00

Наилучшая точность достигается при k=3, k=5 и k=7.
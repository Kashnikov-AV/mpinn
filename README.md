# MPINN - Multi-domain Physics-Informed Neural Networks
Проект разработан для решения задач теплопроводности методом PINN.
## Описание проекта

**MPINN** — это библиотека для решения дифференциальных уравнений в частных производных (PDE) с использованием метода физически информированных нейронных сетей (Physics-Informed Neural Networks, PINN). Библиотека построена на базе **JAX** и **Flax NNX**, что обеспечивает высокую производительность благодаря автоматическому дифференцированию и JIT-компиляции.

### Основные возможности

- **Одномерные задачи теплопроводности** в различных геометриях:
  - Декартова система координат (линейная геометрия)
  - Цилиндрическая система координат
  - Сферическая система координат

- **Типы граничных условий**:
  - Граничные условия Дирихле (заданная температура)
  - Граничные условия Неймана (заданный тепловой поток)
  - Граничные условия Робина (конвективный теплообмен)

- **Многослойные среды (Multi-domain)**:
  - Решение задач для многослойных материалов с различными теплофизическими свойствами
  - Автоматическое обеспечение непрерывности температуры и теплового потока на границах раздела сред

- **Геометрии**:
  - `Interval` — одномерный интервал
  - `Rectangle` — двумерный прямоугольник

### Технологии

- **JAX** — высокопроизводительные численные вычисления
- **Flax NNX** — функциональный API для нейронных сетей
- **Optax** — оптимизаторы для обучения
- **Matplotlib** — визуализация результатов

---

## Структура проекта

```
/workspace/
├── README.md           # Документация проекта
├── pinn_core.py        # Ядро PINN: нейросеть и класс обучения
├── pde.py              # Уравнения в частных производных
├── bc.py               # Граничные условия
├── analytic.py         # Аналитические решения для верификации
├── geom.py             # Геометрические области
├── multidomain.py      # Многослойные задачи (MPINN)
└── docs/               # Подробная документация по модулям
```

---

## Установка зависимостей

```bash
pip install jax jaxlib flax optax matplotlib
```

---

## Быстрый старт

### Пример: Решение одномерного уравнения теплопроводности

```python
import jax.numpy as jnp
import jax.random as random
import flax.nnx as nnx
import optax

from pinn_core import FCNet, PINN
from pde import line_1d
from bc import dirichlet_bc
from analytic import line_1d_dirichlet_exact
from geom import Interval

# Параметры физической задачи
class Physics:
    x_left = 0.0
    x_right = 1.0
    T_left = 300.0  # K
    T_right = 400.0  # K

phys = Physics()

# Создание нейронной сети
rngs = nnx.Rngs(0)
net = FCNet(din=1, dmid=50, dout=1, num_layers=4, 
            activation=nnx.tanh, rngs=rngs)

# Оптимизатор
optimizer = optax.adam(learning_rate=1e-3)

# Веса для компонентов функции потерь
weights = [1.0, 1.0, 1.0]  # [PDE, BC_left, BC_right]

# Инициализация PINN
pinn = PINN(net, optimizer, weights)

# Генерация точек коллокации
geometry = Interval(phys.x_left, phys.x_right)
x_collocation = geometry.generate_collocation(n_interior=100)

# Граничные условия
bc_left = lambda model: dirichlet_bc(model, phys.x_left, phys.T_left)
bc_right = lambda model: dirichlet_bc(model, phys.x_right, phys.T_right)

# Обучение
history, training_time = pinn.fit(
    x_collocation=x_collocation,
    pde_fn=line_1d,
    bc_fns=[bc_left, bc_right],
    phys=phys,
    epochs=10000
)

# Предсказание
x_test = jnp.linspace(phys.x_left, phys.x_right, 100).reshape(-1, 1)
T_pred = pinn.predict(x_test)
T_exact = line_1d_dirichlet_exact(x_test.ravel(), phys)

# Метрики
metrics, _, _ = pinn.evaluate(x_test, line_1d_dirichlet_exact, phys)
print(f"MAE: {metrics['mae']}, RMSE: {metrics['rmse']}")

# Сохранение графика
pinn.save_plot(x_test, T_pred, T_exact, phys, "solution.png")
```

---

## Подробное описание модулей

### 1. `pinn_core.py` — Ядро PINN

Содержит основные классы для построения и обучения PINN:

- **`FCNet`** — полносвязная нейронная сеть
- **`PINN`** — класс для обучения и инференса
- **`normalize/denormalize`** — функции нормализации данных

[Подробнее](docs/pinn_core.md)

### 2. `pde.py` — Дифференциальные уравнения

Реализации операторов PDE для различных геометрий:

- **`line_1d`** — одномерное уравнение в декартовых координатах
- **`cylinder_1d`** — уравнение в цилиндрических координатах
- **`sphere_1d`** — уравнение в сферических координатах

[Подробнее](docs/pde.md)

### 3. `bc.py` — Граничные условия

Функции для вычисления невязок граничных условий:

- **`dirichlet_bc`** — условие Дирихле
- **`neuman_bc`** — условие Неймана
- **`robin_bc`** — условие Робина

[Подробнее](docs/bc.md)

### 4. `analytic.py` — Аналитические решения

Точные решения для верификации численных результатов:

- Для всех комбинаций геометрий и граничных условий
- Используется для расчёта метрик точности

[Подробнее](docs/analytic.md)

### 5. `geom.py` — Геометрия

Классы для генерации точек в расчётной области:

- **`Geometry`** — абстрактный базовый класс
- **`Interval`** — одномерный интервал
- **`Rectangle`** — двумерный прямоугольник

[Подробнее](docs/geom.md)

### 6. `multidomain.py` — Многослойные задачи

Расширение для решения задач в многослойных средах:

- **`MPINN`** — класс для многослойных PINN
- Автоматическое согласование условий на интерфейсах

[Подробнее](docs/multidomain.md)

---

## API Reference

### Класс `FCNet`

```python
FCNet(din, dmid, dout, num_layers, activation, rngs)
```

| Параметр | Описание |
|----------|----------|
| `din` | Размерность входа |
| `dmid` | Размерность скрытых слоёв |
| `dout` | Размерность выхода |
| `num_layers` | Количество скрытых слоёв |
| `activation` | Функция активации |
| `rngs` | Генератор случайных чисел |

### Класс `PINN`

```python
PINN(net, opt, weights)
```

| Метод | Описание |
|-------|----------|
| `fit(x_collocation, pde_fn, bc_fns, phys, epochs)` | Обучение модели |
| `predict(x_test)` | Предсказание на новых данных |
| `evaluate(x_test, exact_fn, phys)` | Оценка метрик |
| `save_plot(...)` | Сохранение графика сравнения |
| `compute_metrics(...)` | Вычисление метрик ошибки |

### Класс `MPINN`

```python
MPINN(nets, opt, weights, phys, n_collocation, rng)
```

| Метод | Описание |
|-------|----------|
| `fit(pde_fn, bc_left_fn, bc_right_fn, phys, epochs)` | Обучение многослойной модели |
| `predict(x_test)` | Предсказание с учётом доменов |
| `evaluate(...)` | Оценка метрик для многослойной задачи |

---

## Метрики качества

Библиотека вычисляет следующие метрики:

| Метрика | Описание |
|---------|----------|
| **MAE** | Средняя абсолютная ошибка |
| **MSE** | Среднеквадратичная ошибка |
| **RMSE** | Корень из MSE |
| **MAPE** | Средняя абсолютная процентная ошибка |
| **Max Error** | Максимальная абсолютная ошибка |

---

## Примеры использования

### Задача с граничным условием Неймана

```python
from bc import neuman_bc

class PhysicsNeuman:
    x_left = 0.0
    x_right = 1.0
    T_left = 300.0
    grad_right = 100.0  # K/м

bc_right = lambda model: neuman_bc(model, phys.x_right, phys.grad_right)
```

### Задача с конвекцией (Робин)

```python
from bc import robin_bc

class PhysicsRobin:
    x_left = 0.0
    x_right = 1.0
    T_left = 300.0
    h = 10.0          # Вт/(м²·K)
    _lambda = 1.0     # Вт/(м·K)
    T_inf = 500.0     # K

bc_right = lambda model: robin_bc(model, phys.x_right, 
                                   phys.alpha, phys.beta, phys.h)
```

### Многослойная стена

```python
from multidomain import MPINN

class PhysicsMulti:
    x_left = 0.0
    x_right = 0.3
    interfaces = [0.1, 0.2]  # Границы между слоями
    all_lambdas = [1.0, 0.5, 2.0]  # Теплопроводности слоёв
    T_left = 300.0
    T_right = 400.0

# Создаём по одной сети на каждый домен
nets = [FCNet(1, 50, 1, 4, nnx.tanh, rngs) for _ in range(3)]
weights = [1.0] * 7  # PDE для 3 доменов + 2 BC + 2 интерфейса

mpinn = MPINN(nets, optax.adam(1e-3), weights, phys)
```

---

## Лицензия

MIT License

---

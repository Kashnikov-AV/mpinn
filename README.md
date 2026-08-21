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

- **Источники тепла**:
  - Постоянные источники
  - Гауссовы источники
  - Точечные источники (аппроксимация дельта-функции)
  - Пользовательские функции источника

- **Геометрии**:
  - `Interval` — одномерный интервал
  - `Rectangle` — двумерный прямоугольник
  - `Circle` — двумерный круг
  - `Box` — трёхмерный параллелепипед

- **Адаптивные стратегии весов**:
  - Фиксированные веса
  - GradNorm (градиентная нормализация)
  - Residual-based (на основе невязок)
  - Uncertainty weighting (гомоскедастическая неопределённость)

### Технологии

- **JAX** — высокопроизводительные численные вычисления
- **Flax NNX** — функциональный API для нейронных сетей
- **Optax** — оптимизаторы для обучения
- **Matplotlib** — визуализация результатов
- **Pandas** — обработка результатов экспериментов

---

## Структура проекта

```
/workspace/
├── README.md                    # Документация проекта
├── main_1d.py                   # Основной скрипт для 1D задач
├── pyproject.toml               # Конфигурация проекта
├── requirements.txt             # Зависимости
├── mpinn/                       # Основной пакет библиотеки
│   ├── __init__.py              # Экспорт модулей
│   ├── pinn_core.py             # Ядро PINN: нейросеть и класс обучения
│   ├── pde.py                   # Уравнения в частных производных
│   ├── bc.py                    # Граничные условия
│   ├── analytic.py              # Аналитические решения для верификации
│   ├── geom.py                  # Геометрические области
│   ├── multidomain.py           # Многослойные задачи (MPINN)
│   ├── config.py                # Конфигурация: PhysicsParams, TrainConfig
│   ├── runner.py                # Запуск экспериментов с Early Stopping
│   ├── plotting.py              # Визуализация результатов
│   ├── sources.py               # Источники тепла
│   └── weight_strategies.py     # Стратегии взвешивания потерь
├── tests/                       # Тесты
│   ├── test_pinn_core.py
│   ├── test_pde.py
│   ├── test_bc.py
│   ├── test_geom.py
│   ├── test_analytic.py
│   └── test_integration.py
├── docs/                        # Подробная документация по модулям
│   ├── pinn_core.md
│   ├── pde.md
│   ├── bc.md
│   ├── analytic.md
│   ├── geom.md
│   └── multidomain.md
└── experiments/                 # Jupyter ноутбуки с примерами
    ├── 1D_dirichlet_*.ipynb
    ├── 1D_neuman_*.ipynb
    └── 1D_robin_*.ipynb
```

---

## Установка зависимостей

```bash
pip install jax jaxlib flax optax matplotlib
```

---

## Быстрый старт

### Пример 1: Решение одномерного уравнения теплопроводности (базовый PINN)

```python
import jax.numpy as jnp
import flax.nnx as nnx
import optax

from mpinn import FCNet, PINN
from mpinn.pde import line_1d
from mpinn.bc import dirichlet_bc
from mpinn.analytic import line_1d_dirichlet_exact
from mpinn.geom import Interval

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
bc_left = lambda model: dirichlet_bc(model, x=phys.x_left, T=phys.T_left)
bc_right = lambda model: dirichlet_bc(model, x=phys.x_right, T=phys.T_right)

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

### Пример 2: Задача с источником тепла

```python
from mpinn import sources
from mpinn.config import PhysicsParams

# Создание источника тепла (гауссов источник)
source_fn = sources.gaussian_source(center=0.5, width=0.1, amplitude=100.0)

# Физические параметры с источником
phys = PhysicsParams(
    x_left=0.0,
    x_right=1.0,
    T_left=300.0,
    T_right=400.0,
    source_fn=source_fn  # Добавляем источник
)

# Использование в PDE
from mpinn.pde import line_1d
# line_1d автоматически учтёт source_fn из phys
```

### Пример 3: Многослойная задача (MPINN)

```python
from mpinn.multidomain import MPINN
from mpinn.weight_strategies import FixedWeightStrategy
from mpinn.config import PhysicsParams

# Параметры для многослойной стены (3 слоя)
phys = PhysicsParams(
    x_left=0.0,
    x_right=0.3,
    interfaces=[0.1, 0.2],  # Границы между слоями
    all_lambdas=[1.0, 0.5, 2.0],  # Теплопроводности слоёв
    T_left=300.0,
    T_right=400.0
)

# Создаём по одной сети на каждый домен
nets = [FCNet(1, 50, 1, 4, nnx.tanh, rngs) for _ in range(3)]

# Стратегия весов
weight_strategy = FixedWeightStrategy()

# Инициализация MPINN
mpinn = MPINN(
    nets=nets,
    opt=optax.adam(1e-3),
    phys=phys,
    n_collocation=100,
    weight_strategy=weight_strategy
)

# Обучение
history, training_time = mpinn.fit(
    pde_fn=line_1d,
    bc_left_fn=lambda m: dirichlet_bc(m, x=phys.x_left, T=phys.T_left),
    bc_right_fn=lambda m: dirichlet_bc(m, x=phys.x_right, T=phys.T_right),
    phys=phys,
    epochs=10000
)
```

### Пример 4: Использование конфигурации и Early Stopping

```python
from mpinn.config import PhysicsParams, TrainConfig, get_activation, get_optimizer
from mpinn.runner import run_experiment

# Конфигурация физики
phys = PhysicsParams(
    x_left=0.0,
    x_right=1.0,
    T_left=300.0,
    T_inf=500.0,
    _lambda=1.0,
    h=10.0
)

# Конфигурация обучения
config = TrainConfig(
    hidden_features=64,
    num_layers=2,
    activation_name='GELU',
    opt_name='adam',
    lr=0.01,
    max_epochs=3000,
    patience=200,  # Early stopping
    num_points=100,
    weights=(1.0, 1.0, 1.0)
)

# Запуск эксперимента
from mpinn.pde import line_1d
from mpinn.analytic import line_1d_robin_exact

result, history = run_experiment(
    config=config,
    phys=phys,
    pde_fn=line_1d,
    exact_fn=line_1d_robin_exact
)

print(f"MSE: {result['mse']}, MAPE: {result['mape']}")
```

---

## Подробное описание модулей

### 1. `mpinn/pinn_core.py` — Ядро PINN

Содержит основные классы для построения и обучения PINN:

- **`FCNet`** — полносвязная нейронная сеть (многослойный перцептрон)
- **`PINN`** — класс для обучения и инференса
- **`normalize/denormalize`** — функции нормализации данных

[Подробнее](docs/pinn_core.md)

**Основные методы PINN:**
- `fit(x_collocation, pde_fn, bc_fns, phys, epochs)` — обучение модели
- `predict(x_test)` — предсказание на новых данных
- `evaluate(x_test, exact_fn, phys)` — оценка метрик (MAE, MSE, RMSE, MAPE, Max Error)
- `save_plot(...)` — сохранение графика сравнения с точным решением
- `compute_metrics(...)` — вычисление метрик ошибки

### 2. `mpinn/pde.py` — Дифференциальные уравнения

Реализации операторов PDE для различных геометрий:

- **`line_1d`** — одномерное уравнение в декартовых координатах: `d²T/dx² + f(x) = 0`
- **`cylinder_1d`** — уравнение в цилиндрических координатах: `dT/dr·(1/r) + d²T/dr² + f(r) = 0`
- **`sphere_1d`** — уравнение в сферических координатах: `d²y/dr² + (2/r)·dy/dr + f(r) = 0`

Все функции поддерживают источники тепла через параметр `phys.source_fn`.

[Подробнее](docs/pde.md)

### 3. `mpinn/bc.py` — Граничные условия

Функции для вычисления невязок граничных условий:

- **`dirichlet_bc(model, x, T)`** — условие Дирихле: `T(x) = T заданное`
- **`neuman_bc(model, x, g)`** — условие Неймана: `dT/dx = g заданное`
- **`robin_bc(model, x, alpha, beta, h)`** — условие Робина: `α·T + β·dT/dx = h`

[Подробнее](docs/bc.md)

### 4. `mpinn/analytic.py` — Аналитические решения

Точные решения для верификации численных результатов:

| Функция | Геометрия | Граничное условие |
|---------|-----------|-------------------|
| `line_1d_dirichlet_exact` | Декартова | Дирихле |
| `line_1d_neuman_exact` | Декартова | Неймана |
| `line_1d_robin_exact` | Декартова | Робина |
| `cylinder_1d_dirichlet_exact` | Цилиндрическая | Дирихле |
| `cylinder_1d_neuman_exact` | Цилиндрическая | Неймана |
| `cylinder_1d_robin_exact` | Цилиндрическая | Робина |
| `sphere_1d_dirichlet_exact` | Сферическая | Дирихле |
| `sphere_1d_neuman_exact` | Сферическая | Неймана |
| `sphere_1d_robin_exact` | Сферическая | Робина |

[Подробнее](docs/analytic.md)

### 5. `mpinn/geom.py` — Геометрия

Классы для генерации точек в расчётной области:

- **`Geometry`** — абстрактный базовый класс
  - `sample_interior(n_points, method, rng)` — точки внутри области
  - `sample_boundary()` — точки на границе
  - `generate_collocation(...)` — все точки коллокации
  - `plot_domain(...)` — визуализация (2D/3D)

- **`Interval(x_left, x_right)`** — одномерный интервал
- **`Rectangle(x_min, x_max, y_min, y_max)`** — двумерный прямоугольник
- **`Circle(cx, cy, radius)`** — двумерный круг
- **`Box(x_min, x_max, y_min, y_max, z_min, z_max)`** — трёхмерный параллелепипед

Методы дискретизации: `'random'` (случайная), `'uniform'` (равномерная сетка).

[Подробнее](docs/geom.md)

### 6. `mpinn/multidomain.py` — Многослойные задачи (MPINN)

Расширение для решения задач в многослойных средах:

- **`MPINN`** — класс для многослойных PINN
  - Координирует несколько независимых PINN (по одной на домен)
  - Обеспечивает непрерывность температуры и потока на интерфейсах
  - Поддерживает адаптивные стратегии весов

**Условия на интерфейсах:**
- Непрерывность температуры: `(T_left - T_right)²`
- Непрерывность потока: `(λ_left·dT/dx|left - λ_right·dT/dx|right)²`

[Подробнее](docs/multidomain.md)

### 7. `mpinn/config.py` — Конфигурация

Классы данных для конфигурации экспериментов:

- **`PhysicsParams`** — физические параметры задачи:
  - `x_left`, `x_right` — границы области
  - `T_left`, `T_right` — температуры на границах
  - `_lambda` — теплопроводность
  - `h` — коэффициент конвекции
  - `T_inf` — температура окружающей среды
  - `source_fn` — функция источника тепла
  - `interfaces`, `all_lambdas` — для многослойных задач

- **`TrainConfig`** — гиперпараметры обучения:
  - `hidden_features`, `num_layers` — архитектура сети
  - `activation_name` — функция активации
  - `opt_name`, `lr` — оптимизатор и скорость обучения
  - `max_epochs`, `patience`, `min_delta` — параметры Early Stopping
  - `weights` — веса компонентов функции потерь

- **`get_activation(name)`** — фабрика функций активации
- **`get_optimizer(name, lr)`** — фабрика оптимизаторов

### 8. `mpinn/sources.py` — Источники тепла

Фабрики функций источников тепла для уравнения `-d/dx(k·dT/dx) = f(x)`:

- **`constant_source(value)`** — постоянный источник: `f(x) = value`
- **`gaussian_source(center, width, amplitude)`** — гауссов источник
- **`point_source(location, epsilon, amplitude)`** — точечный источник (аппроксимация дельта-функции)
- **`custom_source(func)`** — пользовательская функция
- **`no_source()`** — нулевой источник (однородное уравнение)

### 9. `mpinn/weight_strategies.py` — Стратегии взвешивания

Базовый класс и реализации стратегий для балансировки компонентов функции потерь:

- **`BaseWeightStrategy`** — абстрактный базовый класс
  - `compute_weights(losses, step, model_state)` — вычисление весов
  - `reset()` — сброс состояния

- **`FixedWeightStrategy`** — фиксированные веса (по умолчанию)
- **`GradNormStrategy`** — градиентная нормализация (placeholder)
- **`ResidualBasedStrategy`** — на основе невязок (placeholder)
- **`UncertaintyWeightingStrategy`** — гомоскедастическая неопределённость (placeholder)

### 10. `mpinn/runner.py` — Запуск экспериментов

Утилиты для проведения экспериментов:

- **`run_experiment(config, phys, pde_fn, exact_fn, ...)`** — запуск одного эксперимента с Early Stopping
- **`run_grid_search(param_grid, base_config, phys, ...)`** — перебор комбинаций гиперпараметров
- **`_train_with_early_stopping(...)`** — обучение с ранней остановкой

Возвращаемые метрики: MSE, MAE, RMSE, MAPE, Max Error, время обучения.

### 11. `mpinn/plotting.py` — Визуализация

Функции для отрисовки результатов:

- **`show_plot(x_test, T_pred, T_exact, phys, title)`** — отображение сравнения решений
- **`save_plot(x_test, T_pred, T_exact, phys, save_path, title)`** — сохранение графика
- **`show_history(history, save_path, show_plot)`** — график истории обучения (потери по эпохам)

---

## API Reference

### Класс `FCNet`

```python
FCNet(din, dmid, dout, num_layers, activation, rngs)
```

| Параметр | Описание |
|----------|----------|
| `din` | Размерность входа (для 1D задач = 1) |
| `dmid` | Размерность скрытых слоёв |
| `dout` | Размерность выхода (для задач теплопроводности = 1) |
| `num_layers` | Количество скрытых слоёв |
| `activation` | Функция активации (`nnx.tanh`, `nnx.gelu`, `nnx.relu`, и т.д.) |
| `rngs` | Генератор случайных чисел `nnx.Rngs(seed)` |

### Класс `PINN`

```python
PINN(net, opt, weights)
```

| Параметр | Описание |
|----------|----------|
| `net` | Экземпляр FCNet |
| `opt` | Оптимизатор Optax (например, `optax.adam(lr)`) |
| `weights` | Список весов [w_pde, w_bc0, w_bc1, ...] |

| Метод | Описание |
|-------|----------|
| `fit(x_collocation, pde_fn, bc_fns, phys, epochs)` | Обучение модели |
| `predict(x_test)` | Предсказание на новых данных |
| `evaluate(x_test, exact_fn, phys, bc_names)` | Оценка метрик |
| `save_plot(x_test, T_pred, T_exact, phys, save_path)` | Сохранение графика сравнения |
| `show_plot(x_test, T_pred, T_exact, phys)` | Отображение графика |
| `compute_metrics(x_test, T_pred, T_exact)` | Вычисление метрик ошибки |

### Класс `MPINN`

```python
MPINN(nets, opt, phys, n_collocation, weight_strategy, rng)
```

| Параметр | Описание |
|----------|----------|
| `nets` | Кортеж экземпляров FCNet (по одному на домен) |
| `opt` | Оптимизатор Optax |
| `phys` | PhysicsParams с полями `interfaces` и `all_lambdas` |
| `n_collocation` | Число точек коллокации на домен |
| `weight_strategy` | Стратегия взвешивания (по умолчанию `FixedWeightStrategy()`) |
| `rng` | JAX random key |

| Метод | Описание |
|-------|----------|
| `fit(pde_fn, bc_left_fn, bc_right_fn, phys, epochs, log_interval)` | Обучение многослойной модели |
| `predict(x_test)` | Предсказание с учётом доменов |
| `evaluate(x_test, exact_fn, phys, bc_names)` | Оценка метрик для многослойной задачи |
| `save_plot(...)` / `show_plot(...)` | Визуализация с маркерами интерфейсов |

---

## Метрики качества

Библиотека вычисляет следующие метрики:

| Метрика | Формула | Описание |
|---------|---------|----------|
| **MAE** | `mean(|T_pred - T_exact|)` | Средняя абсолютная ошибка |
| **MSE** | `mean((T_pred - T_exact)²)` | Среднеквадратичная ошибка |
| **RMSE** | `sqrt(MSE)` | Корень из MSE |
| **MAPE** | `mean(|(T_pred - T_exact)/T_exact|)` | Средняя абсолютная процентная ошибка |
| **Max Error** | `max(|T_pred - T_exact|)` | Максимальная абсолютная ошибка |

---

## Примеры использования

### Задача с граничным условием Неймана

```python
from mpinn.bc import neuman_bc
from mpinn.config import PhysicsParams

class PhysicsNeuman:
    x_left = 0.0
    x_right = 1.0
    T_left = 300.0
    grad_right = 100.0  # K/м

phys = PhysicsNeuman()
bc_right = lambda model: neuman_bc(model, x=phys.x_right, g=phys.grad_right)
```

### Задача с конвекцией (Робин)

```python
from mpinn.bc import robin_bc
from mpinn.config import PhysicsParams

phys = PhysicsParams(
    x_left=0.0,
    x_right=1.0,
    T_left=300.0,
    h=10.0,          # Вт/(м²·K)
    _lambda=1.0,     # Вт/(м·K)
    T_inf=500.0      # K
)

# Условие Робина: α·T + β·dT/dx = γ
# где α=h, β=λ, γ=h·T_inf
bc_right = lambda model: robin_bc(
    model, 
    x=phys.x_right, 
    alpha=phys.h, 
    beta=phys._lambda, 
    h=phys.h * phys.T_inf
)
```

### Многослойная стена (3 слоя)

```python
from mpinn.multidomain import MPINN
from mpinn.weight_strategies import FixedWeightStrategy
from mpinn.config import PhysicsParams

# Параметры для трёхслойной стены
phys = PhysicsParams(
    x_left=0.0,
    x_right=0.3,
    interfaces=[0.1, 0.2],        # Границы между слоями
    all_lambdas=[1.0, 0.5, 2.0],  # Теплопроводности слоёв
    T_left=300.0,
    T_right=400.0
)

# Создаём по одной сети на каждый домен
nets = tuple(FCNet(1, 50, 1, 4, nnx.tanh, rngs) for _ in range(3))

# Стратегия весов
weight_strategy = FixedWeightStrategy()

# Инициализация MPINN
mpinn = MPINN(
    nets=nets,
    opt=optax.adam(1e-3),
    phys=phys,
    n_collocation=100,
    weight_strategy=weight_strategy
)

# Веса потерь: PDE для каждого домена + BC слева + BC справа + интерфейсы
# Всего: 3 (PDE) + 2 (BC) + 2 (interface) = 7 компонентов
```

### Grid Search гиперпараметров

```python
from mpinn.runner import run_grid_search
from mpinn.config import TrainConfig, PhysicsParams

param_grid = {
    'hidden_features': [32, 64, 128],
    'num_layers': [2, 3, 4],
    'lr': [0.001, 0.01, 0.1],
    'activation_name': ['tanh', 'GELU', 'silu']
}

base_config = TrainConfig()
phys = PhysicsParams(x_left=0.0, x_right=1.0, T_left=300.0, T_right=400.0)

df_results = run_grid_search(
    param_grid=param_grid,
    base_config=base_config,
    phys=phys,
    csv_path='results/grid_search.csv'
)

# Поиск лучшей конфигурации
best_idx = df_results['mse'].idxmin()
best_config = df_results.loc[best_idx]
print(f"Лучшая конфигурация: {best_config}")
```

---

## Лицензия

MIT License

---

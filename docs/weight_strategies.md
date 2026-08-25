# Модуль weight_strategies.py

## Обзор

Модуль `weight_strategies.py` предоставляет стратегии для управления весами потерь в многодоменных PINN (MPINN). Стратегии определяют, как балансировать различные компоненты функции потерь (PDE, граничные условия, условия на интерфейсах) во время обучения.

## Зависимости

- `abc.ABC` — абстрактные базовые классы
- `typing` — аннотации типов
- `jax.numpy` — численные операции
- `flax.nnx` — нейронные сети

---

## Базовый класс

### BaseWeightStrategy

Абстрактный базовый класс для всех стратегий взвешивания потерь.

#### Конструктор

```python
BaseWeightStrategy(initial_weights: Optional[Dict[str, float]] = None)
```

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `initial_weights` | Optional[Dict[str, float]] | Словарь начальных весов для компонентов потерь |

#### Метод compute_weights (абстрактный)

```python
@abstractmethod
def compute_weights(
    losses: Dict[str, Dict[str, float]],
    step: int,
    model_state: Optional[Any] = None
) -> Dict[str, Dict[str, float]]
```

Вычисляет веса для текущего шага обучения.

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `losses` | Dict[str, Dict[str, float]] | Вложенный словарь потерей {domain: {loss_type: value}} |
| `step` | int | Текущий шаг обучения |
| `model_state` | Optional[Any] | Состояние модели (параметры, градиенты) |

**Возвращает:**

- `Dict[str, Dict[str, float]]` — веса {domain: {loss_type: weight}}

#### Метод reset

```python
def reset()
```

Сбрасывает внутреннее состояние стратегии.

---

## Реализации стратегий

### FixedWeightStrategy

Стратегия с фиксированными, заданными пользователем весами. Это поведение по умолчанию.

#### Конструктор

```python
FixedWeightStrategy(initial_weights: Optional[Dict[str, float]] = None)
```

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `initial_weights` | Optional[Dict[str, float]] | Начальные веса с ключами вида 'pde_pde_0', 'bc_bc_left', 'interface_interface_0' |

#### Пример использования

```python
from mpinn.weight_strategies import FixedWeightStrategy
from mpinn.multidomain import MPINN

# Стратегия с фиксированными весами
strategy = FixedWeightStrategy(
    initial_weights={
        'pde_pde_0': 1.0,
        'pde_pde_1': 1.0,
        'bc_bc_left': 2.0,
        'bc_bc_right': 2.0,
        'interface_interface_0': 5.0  # Усилить условие на интерфейсе
    }
)

mpinn = MPINN(
    nets=nets,
    opt=optimizer,
    phys=phys,
    n_collocation=100,
    weight_strategy=strategy
)
```

---

### GradNormStrategy (заготовка)

Стратегия нормализации градиентов (GradNorm). Динамически调整рует веса для балансировки норм градиентов across задач.

Ссылка: Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks"

#### Статус

⚠️ **Не реализована** — требуется доработка.

#### Планируемый API

```python
from mpinn.weight_strategies import GradNormStrategy

strategy = GradNormStrategy(
    initial_weights={'pde': 1.0, 'bc': 1.0, 'interface': 1.0}
)

mpinn = MPINN(nets, opt, phys, weight_strategy=strategy)
```

---

### ResidualBasedStrategy (заготовка)

Стратегия адаптивного взвешивания на основе невязок. Увеличивает веса для компонентов с большими невязками.

#### Статус

⚠️ **Не реализована** — требуется доработка.

#### Планируемый API

```python
from mpinn.weight_strategies import ResidualBasedStrategy

strategy = ResidualBasedStrategy(
    initial_weights={'pde': 1.0, 'bc': 1.0, 'interface': 1.0}
)
```

---

### UncertaintyWeightingStrategy (заготовка)

Стратегия взвешивания на основе гомоскедастической неопределённости. Обучает параметры логарифмической дисперсии для автоматического балансирования потерь.

Ссылка: Kendall & Gal, "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"

#### Статус

⚠️ **Не реализована** — требуется доработка.

#### Планируемый API

```python
from mpinn.weight_strategies import UncertaintyWeightingStrategy

strategy = UncertaintyWeightingStrategy(
    initial_weights={'pde': 1.0, 'bc': 1.0, 'interface': 1.0}
)
```

---

## Формат словаря потерь

Стратегии получают словарь потерь следующей структуры:

```python
losses = {
    'pde': {
        'pde_0': 0.001,      # PDE потеря для домена 0
        'pde_1': 0.002,      # PDE потеря для домена 1
        # ...
    },
    'bc': {
        'bc_left': 0.005,    # Потеря левого ГК
        'bc_right': 0.003,   # Потеря правого ГК
    },
    'interface': {
        'interface_0': 0.01, # Потеря на интерфейсе 0
        'interface_1': 0.02, # Потеря на интерфейсе 1
        # ...
    }
}
```

---

## Формат возвращаемых весов

Метод `compute_weights` возвращает веса в аналогичной структуре:

```python
weights = {
    'pde': {
        'pde_0': 1.0,
        'pde_1': 1.0,
    },
    'bc': {
        'bc_left': 2.0,
        'bc_right': 2.0,
    },
    'interface': {
        'interface_0': 5.0,
        'interface_1': 5.0,
    }
}
```

---

## Полный пример использования

```python
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from mpinn.config import PhysicsParams
from mpinn.pinn_core import FCNet
from mpinn.multidomain import MPINN
from mpinn.weight_strategies import FixedWeightStrategy

# 1. Физические параметры для двух доменов
phys = PhysicsParams(
    x_left=0.0,
    x_right=2.0,
    T_left=300.0,
    T_right=400.0,
    _lambda=1.0,
    h=10.0,
    interfaces=[1.0],           # Интерфейс между доменами
    all_lambdas=[1.0, 2.0]      # Теплопроводности доменов
)

# 2. Создание сетей для каждого домена
nets = tuple(
    FCNet(
        din=1,
        dmid=64,
        dout=1,
        num_layers=2,
        activation=nnx.gelu,
        rngs=nnx.Rngs(i)
    )
    for i in range(2)
)

# 3. Оптимизатор
optimizer = optax.adam(0.01)

# 4. Стратегия взвешивания
strategy = FixedWeightStrategy(
    initial_weights={
        'pde_pde_0': 1.0,
        'pde_pde_1': 1.0,
        'bc_bc_left': 1.0,
        'bc_bc_right': 1.0,
        'interface_interface_0': 10.0  # Усилить условие непрерывности
    }
)

# 5. Создание MPINN
mpinn = MPINN(
    nets=nets,
    opt=optimizer,
    phys=phys,
    n_collocation=100,
    weight_strategy=strategy
)

# 6. Обучение
from mpinn.pde import line_1d
from mpinn.bc import dirichlet_bc

bc_left_fn = lambda m: dirichlet_bc(m, jnp.array([[phys.x_left]]), phys.T_left)
bc_right_fn = lambda m: dirichlet_bc(m, jnp.array([[phys.x_right]]), phys.T_right)

history, training_time = mpinn.fit(
    pde_fn=line_1d,
    bc_left_fn=bc_left_fn,
    bc_right_fn=bc_right_fn,
    phys=phys,
    epochs=5000
)

print(f"Обучение завершено за {training_time:.2f}c")
print(f"PDE потери домена 0: {history['pde_0'][-1]:.6e}")
print(f"PDE потери домена 1: {history['pde_1'][-1]:.6e}")
print(f"Потеря на интерфейсе: {history['interface_0'][-1]:.6e}")
```

---

## Рекомендации по выбору весов

### Для FixedWeightStrategy

Типичные значения весов:

```python
initial_weights = {
    'pde_pde_0': 1.0,          # Базовый вес PDE
    'pde_pde_1': 1.0,
    'bc_bc_left': 1.0,         # Граничные условия
    'bc_bc_right': 1.0,
    'interface_interface_0': 5.0  # Условия на интерфейсе (обычно выше)
}
```

**Рекомендации:**

- **PDE**: 0.1 – 10.0 (зависит от масштаба невязок)
- **Граничные условия**: 1.0 – 10.0
- **Интерфейсы**: 5.0 – 50.0 (требуют больших весов для точного соблюдения)

### Адаптивная настройка весов

Если обучение нестабильно:

1. Увеличьте вес компонента с большой потерей
2. Используйте логарифмический масштаб для подбора
3. Мониторьте соотношение потерь во время обучения

```python
# Пример настройки при проблемах с интерфейсом
strategy = FixedWeightStrategy(
    initial_weights={
        'pde_pde_0': 0.1,      # Уменьшить PDE
        'pde_pde_1': 0.1,
        'bc_bc_left': 1.0,
        'bc_bc_right': 1.0,
        'interface_interface_0': 20.0  # Усилить интерфейс
    }
)
```

---

## Создание собственной стратегии

Для реализации адаптивной стратегии унаследуйтесь от `BaseWeightStrategy`:

```python
from mpinn.weight_strategies import BaseWeightStrategy
from typing import Dict, Any, Optional

class MyAdaptiveStrategy(BaseWeightStrategy):
    """Адаптивная стратегия на основе отношения потерь."""
    
    def compute_weights(
        self,
        losses: Dict[str, Dict[str, float]],
        step: int,
        model_state: Optional[Any] = None
    ) -> Dict[str, Dict[str, float]]:
        # Ваша логика вычисления весов
        weights = {}
        
        for domain, domain_losses in losses.items():
            weights[domain] = {}
            total_loss = sum(domain_losses.values())
            
            for loss_name, loss_value in domain_losses.items():
                # Адаптивный вес: обратно пропорционален потере
                if loss_value > 0:
                    weight = 1.0 / (loss_value + 1e-8)
                else:
                    weight = 1.0
                weights[domain][loss_name] = weight
        
        return weights

# Использование
strategy = MyAdaptiveStrategy()
mpinn = MPINN(nets, opt, phys, weight_strategy=strategy)
```

---

## См. также

- [multidomain.md](multidomain.md) — Многодоменные задачи (MPINN)
- [pinn_core.md](pinn_core.md) — Ядро PINN
- [config.md](config.md) — Конфигурация

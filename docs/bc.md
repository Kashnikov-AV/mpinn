# Модуль bc.py

## Обзор

Модуль `bc.py` содержит реализации функций для вычисления невязок граничных условий (Boundary Conditions, BC). Граничные условия необходимы для однозначной постановки задачи и обеспечения физического поведения решения на границах расчётной области.

**Важное изменение:** Все функции теперь вычисляют **среднюю арифметическую невязку** по всем точкам границы (1/N * Σ), что обеспечивает корректное усреднение потерь при работе с несколькими граничными точками.

## Зависимости

- `jax` — автоматическое дифференцирование
- `jax.numpy` — численные операции

---

## Типы граничных условий

### 1. Условие Дирихле (Dirichlet)

Заданное значение искомой величины на границе (например, температура).

**Формула:**

```
T(x_boundary) = T_given
```

**Невязка (среднее арифметическое):**

```
loss = (1/N) * Σ(T_pred(xi) - T_given)²
```

### 2. Условие Неймана (Neumann)

Заданная производная искомой величины на границе (например, тепловой поток).

**Формула:**

```
dT/dx(x_boundary) = g_given
```

**Невязка (среднее арифметическое):**

```
loss = (1/N) * Σ(dT/dx_pred(xi) - g_given)²
```

### 3. Условие Робина (Robin)

Линейная комбинация значения и производной на границе (конвективный теплообмен).

**Формула:**

```
α·T + β·dT/dx = h
```

Для задачи теплопроводности с конвекцией:

```
h_conv · (T_inf - T_surface) = -λ · dT/dx
```

**Невязка (среднее арифметическое):**

```
loss = (1/N) * Σ(α·T_pred(xi) + β·dT/dx_pred(xi) - h)²
```

---

## Функции

### dirichlet_bc

Граничное условие Дирихле. Вычисляет среднюю квадратичную невязку по всем точкам границы.

**Сигнатура:**

```python
dirichlet_bc(model, x, T)
```

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `model` | nnx.Module | Нейронная сеть |
| `x` | jax.Array | Массив координат граничных точек |
| `T` | float или jax.Array | Заданное значение температуры |

**Возвращает:**

- `float` — средняя квадратичная невязка `(1/N) * Σ(T_pred - T)²`

**Пример:**

```python
from bc import dirichlet_bc
import jax.numpy as jnp

# Левая граница: T = 300 K (несколько точек)
x_left = jnp.array([0.0, 0.01, 0.02])
bc_left = lambda model: dirichlet_bc(model, x=x_left, T=300.0)

# Правая граница: T = 400 K
x_right = jnp.array([1.0, 0.99, 0.98])
bc_right = lambda model: dirichlet_bc(model, x=x_right, T=400.0)
```

---

### neuman_bc

Граничное условие Неймана. Вычисляет среднюю квадратичную невязку по всем точкам границы.

**Сигнатура:**

```python
neuman_bc(model, x, g)
```

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `model` | nnx.Module | Нейронная сеть |
| `x` | jax.Array | Массив координат граничных точек |
| `g` | float или jax.Array | Заданное значение производной (градиент) |

**Возвращает:**

- `float` — средняя квадратичная невязка `(1/N) * Σ(dT/dx_pred - g)²`

**Пример:**

```python
from bc import neuman_bc
import jax.numpy as jnp

# Теплоизолированная граница: dT/dx = 0 (несколько точек)
x_left = jnp.array([0.0, 0.01, 0.02])
bc_insulated = lambda model: neuman_bc(model, x=x_left, g=0.0)

# Заданный тепловой поток: dT/dx = 100 K/м
x_right = jnp.array([1.0, 0.99, 0.98])
bc_flux = lambda model: neuman_bc(model, x=x_right, g=100.0)
```

---

### robin_bc

Граничное условие Робина (конвекция). Вычисляет среднюю квадратичную невязку по всем точкам границы.

**Сигнатура:**

```python
robin_bc(model, x, alpha, beta, h)
```

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `model` | nnx.Module | Нейронная сеть |
| `x` | jax.Array | Массив координат граничных точек |
| `alpha` | float | Коэффициент при T |
| `beta` | float | Коэффициент при dT/dx |
| `h` | float | Свободный член (обычно h_conv · T_inf) |

**Возвращает:**

- `float` — средняя квадратичная невязка `(1/N) * Σ(α·T + β·dT/dx - h)²`

**Пример:**

```python
from bc import robin_bc
import jax.numpy as jnp

# Конвекция на правой границе (несколько точек):
# h_conv * (T_inf - T) = -lambda * dT/dx
# Перепишем: h_conv * T + lambda * dT/dx = h_conv * T_inf

h_conv = 10.0      # Вт/(м²·K)
_lambda = 1.0      # Вт/(м·K)
T_inf = 500.0      # K

x_right = jnp.array([1.0, 0.99, 0.98])
bc_convection = lambda model: robin_bc(
    model, 
    x=x_right,
    alpha=h_conv, 
    beta=_lambda, 
    h=h_conv * T_inf
)
```

---

## Использование в PINN

Граничные условия передаются в метод `fit()` как список функций:

```python
from pinn_core import PINN
from bc import dirichlet_bc, neuman_bc
import jax.numpy as jnp

# Создание PINN
pinn = PINN(net, optimizer, weights=[1.0, 1.0, 1.0])

# Определение граничных условий (с несколькими точками)
x_left = jnp.linspace(phys.x_left, phys.x_left + 0.01, 5)
x_right = jnp.linspace(phys.x_right - 0.01, phys.x_right, 5)

bc_left = lambda m: dirichlet_bc(m, x=x_left, T=phys.T_left)
bc_right = lambda m: neuman_bc(m, x=x_right, g=phys.grad_right)

# Обучение
history, time = pinn.fit(
    x_collocation=x_points,
    pde_fn=line_1d,
    bc_fns=[bc_left, bc_right],  # Список BC
    phys=phys,
    epochs=10000
)
```

---

## Примеры использования

### Задача с двумя условиями Дирихле

```python
import jax.numpy as jnp

class Physics:
    x_left = 0.0
    x_right = 1.0
    T_left = 300.0
    T_right = 400.0

phys = Physics()

# Несколько точек на каждой границе для лучшего обучения
x_left = jnp.linspace(phys.x_left, phys.x_left + 0.01, 5)
x_right = jnp.linspace(phys.x_right - 0.01, phys.x_right, 5)

bc_left = lambda m: dirichlet_bc(m, x=x_left, T=phys.T_left)
bc_right = lambda m: dirichlet_bc(m, x=x_right, T=phys.T_right)

pinn.fit(..., bc_fns=[bc_left, bc_right], ...)
```

### Задача Дирихле-Неймана

```python
class Physics:
    x_left = 0.0
    x_right = 1.0
    T_left = 300.0
    grad_right = 0.0  # Теплоизолированная граница

phys = Physics()

x_left = jnp.linspace(phys.x_left, phys.x_left + 0.01, 5)
x_right = jnp.linspace(phys.x_right - 0.01, phys.x_right, 5)

bc_left = lambda m: dirichlet_bc(m, x=x_left, T=phys.T_left)
bc_right = lambda m: neuman_bc(m, x=x_right, g=phys.grad_right)

pinn.fit(..., bc_fns=[bc_left, bc_right], ...)
```

### Задача с конвекцией (Робин)

```python
class Physics:
    x_left = 0.0
    x_right = 1.0
    T_left = 300.0
    h_conv = 10.0       # Вт/(м²·K)
    _lambda = 1.0       # Вт/(м·K)
    T_inf = 500.0       # K

phys = Physics()

x_left = jnp.linspace(phys.x_left, phys.x_left + 0.01, 5)
x_right = jnp.linspace(phys.x_right - 0.01, phys.x_right, 5)

bc_left = lambda m: dirichlet_bc(m, x=x_left, T=phys.T_left)
bc_right = lambda m: robin_bc(
    m, 
    x=x_right,
    alpha=phys.h_conv,
    beta=phys._lambda,
    h=phys.h_conv * phys.T_inf
)

pinn.fit(..., bc_fns=[bc_left, bc_right], ...)
```

---

## Математические основы

### Автоматическое дифференцирование

Для вычисления производных в условиях Неймана и Робина используется автоматическое дифференцирование JAX:

```python
def get_T(x_val):
    return model(jnp.array([[x_val]])).ravel()[0]

dT_dx = jax.grad(get_T)(xi)  # Для каждой точки xi
```

Это обеспечивает точное вычисление градиентов без численных погрешностей.

### Усреднение невязок

**Ключевое изменение:** Все функции теперь используют среднее арифметическое вместо суммы:

```python
# Старый подход (сумма):
loss = sum(residuals)

# Новый подход (среднее арифметическое):
residuals = jnp.array([compute_residual(xi) for xi in x])
loss = jnp.mean(residuals)  # (1/N) * Σ residuals
```

Это обеспечивает:
- **Масштабируемость**: потери не зависят от количества граничных точек
- **Стабильность обучения**: градиенты имеют сопоставимый масштаб при разном числе точек
- **Интерпретируемость**: значение потерь представляет средний квадрат ошибки на одну точку

### Веса граничных условий

Веса для компонентов граничных условий задаются в конструкторе `PINN`:

```python
# weights = [w_pde, w_bc_left, w_bc_right]
pinn = PINN(net, optimizer, weights=[1.0, 1.0, 1.0])
```

Общая функция потерь:

```
L_total = w_pde · L_pde + w_bc_left · L_bc_left + w_bc_right · L_bc_right
```

где каждое `L_bc` уже является усреднённой невязкой.

---

## См. также

- [pinn_core.md](pinn_core.md) — Ядро PINN
- [pde.md](pde.md) — Дифференциальные уравнения
- [analytic.md](analytic.md) — Аналитические решения

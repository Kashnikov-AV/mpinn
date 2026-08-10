# Модуль bc.py

## Обзор

Модуль `bc.py` содержит реализации функций для вычисления невязок граничных условий (Boundary Conditions, BC). Граничные условия необходимы для однозначной постановки задачи и обеспечения физического поведения решения на границах расчётной области.

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

**Невязка:**

```
loss = (T_pred - T_given)²
```

### 2. Условие Неймана (Neumann)

Заданная производная искомой величины на границе (например, тепловой поток).

**Формула:**

```
dT/dx(x_boundary) = g_given
```

**Невязка:**

```
loss = (dT/dx_pred - g_given)²
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

**Невязка:**

```
loss = (α·T_pred + β·dT/dx_pred - h)²
```

---

## Функции

### dirichlet_bc

Граничное условие Дирихле.

**Сигнатура:**

```python
dirichlet_bc(model, x, T)
```

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `model` | nnx.Module | Нейронная сеть |
| `x` | float | Координата границы |
| `T` | float | Заданное значение температуры |

**Возвращает:**

- `jax.Array` — квадрат невязки `(T_pred - T)²`

**Пример:**

```python
from bc import dirichlet_bc

# Левая граница: T = 300 K
bc_left = lambda model: dirichlet_bc(model, x=0.0, T=300.0)

# Правая граница: T = 400 K
bc_right = lambda model: dirichlet_bc(model, x=1.0, T=400.0)
```

---

### neuman_bc

Граничное условие Неймана.

**Сигнатура:**

```python
neuman_bc(model, x, g)
```

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `model` | nnx.Module | Нейронная сеть |
| `x` | float | Координата границы |
| `g` | float | Заданное значение производной (градиент) |

**Возвращает:**

- `jax.Array` — квадрат невязки `(dT/dx_pred - g)²`

**Пример:**

```python
from bc import neuman_bc

# Теплоизолированная граница: dT/dx = 0
bc_insulated = lambda model: neuman_bc(model, x=0.0, g=0.0)

# Заданный тепловой поток: dT/dx = 100 K/м
bc_flux = lambda model: neuman_bc(model, x=1.0, g=100.0)
```

---

### robin_bc

Граничное условие Робина (конвекция).

**Сигнатура:**

```python
robin_bc(model, x, alpha, beta, h)
```

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `model` | nnx.Module | Нейронная сеть |
| `x` | float | Координата границы |
| `alpha` | float | Коэффициент при T |
| `beta` | float | Коэффициент при dT/dx |
| `h` | float | Свободный член (обычно h_conv · T_inf) |

**Возвращает:**

- `jax.Array` — квадрат невязки `(α·T + β·dT/dx - h)²`

**Пример:**

```python
from bc import robin_bc

# Конвекция на правой границе:
# h_conv * (T_inf - T) = -lambda * dT/dx
# Перепишем: h_conv * T + lambda * dT/dx = h_conv * T_inf

h_conv = 10.0      # Вт/(м²·K)
_lambda = 1.0      # Вт/(м·K)
T_inf = 500.0      # K

bc_convection = lambda model: robin_bc(
    model, 
    x=1.0, 
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

# Создание PINN
pinn = PINN(net, optimizer, weights=[1.0, 1.0, 1.0])

# Определение граничных условий
bc_left = lambda m: dirichlet_bc(m, x=phys.x_left, T=phys.T_left)
bc_right = lambda m: neuman_bc(m, x=phys.x_right, g=phys.grad_right)

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
class Physics:
    x_left = 0.0
    x_right = 1.0
    T_left = 300.0
    T_right = 400.0

phys = Physics()

bc_left = lambda m: dirichlet_bc(m, phys.x_left, phys.T_left)
bc_right = lambda m: dirichlet_bc(m, phys.x_right, phys.T_right)

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

bc_left = lambda m: dirichlet_bc(m, phys.x_left, phys.T_left)
bc_right = lambda m: neuman_bc(m, phys.x_right, phys.grad_right)

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

bc_left = lambda m: dirichlet_bc(m, phys.x_left, phys.T_left)
bc_right = lambda m: robin_bc(
    m, 
    phys.x_right,
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

dT_dx = jax.grad(get_T)(x)
```

Это обеспечивает точное вычисление градиентов без численных погрешностей.

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

---

## См. также

- [pinn_core.md](pinn_core.md) — Ядро PINN
- [pde.md](pde.md) — Дифференциальные уравнения
- [analytic.md](analytic.md) — Аналитические решения

# Метод Главных Компонент (PCA): Математический Вывод

## 1. Постановка задачи и Центрирование
$X \in \mathbb{R}^{n \times m}, \quad x_i \in \mathbb{R}^m \quad (i = 1, \dots, n)$

$$ \mu = \frac{1}{n} \sum_{i=1}^{n} x_i $$

$$ x_i \leftarrow x_i - \mu \quad \implies \quad \sum_{i=1}^{n} x_i = \mathbf{0} $$

---

## 2. Геометрия проекции и Ошибка реконструкции (Теорема Пифагора)
Проекция на направляющий вектор $\mathbf{w} \in \mathbb{R}^m \quad (\|\mathbf{w}\|_2 = 1 \iff \mathbf{w}^T\mathbf{w} = 1)$:
- Скалярная проекция: $x_i \mathbf{w}$
- Вектор проекции: $(x_i \mathbf{w}) \mathbf{w}^T$
- Вектор ошибки (потеря информации): $e_i = x_i - (x_i \mathbf{w}) \mathbf{w}^T$

$$ \|x_i\|^2 = \|(x_i \mathbf{w}) \mathbf{w}^T\|^2 + \|e_i\|^2 = (x_i \mathbf{w})^2 \|\mathbf{w}^T\|^2 + \|e_i\|^2 = (x_i \mathbf{w})^2 + \|e_i\|^2 $$

Разделив сумму по $n$ наблюдениям:

$$ \frac{1}{n} \sum_{i=1}^n \|x_i\|^2 = \frac{1}{n} \sum_{i=1}^n (x_i \mathbf{w})^2 + \frac{1}{n} \sum_{i=1}^n \|e_i\|^2 $$

$$ \text{TotalVar} = \text{Var}(\text{Proj}_{\mathbf{w}}) + \text{MSE}_{\mathbf{w}} $$

$$ \max_{\mathbf{w}} \text{Var}(\text{Proj}_{\mathbf{w}}) \iff \min_{\mathbf{w}} \text{MSE}_{\mathbf{w}} $$

---

## 3. Задача Оптимизации
Дисперсия проекции:

$$ \text{Var}(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^n (x_i \mathbf{w})^2 = \frac{1}{n} (X\mathbf{w})^T (X\mathbf{w}) = \frac{1}{n} (\mathbf{w}^T X^T) (X\mathbf{w}) $$

$$ \text{Var}(\mathbf{w}) = \mathbf{w}^T \left(\frac{1}{n} X^T X\right) \mathbf{w} = \mathbf{w}^T C \mathbf{w} $$

Ковариационная матрица: $C = \frac{1}{n} X^T X, \quad C = C^T, \quad C \in \mathbb{R}^{m \times m}$

**Максимизация дисперсии при условии единичной нормы (чтобы найти направление):**

$$ \max_{\mathbf{w}} \mathbf{w}^T C \mathbf{w} \quad \text{s.t.} \quad \mathbf{w}^T \mathbf{w} = 1 $$

---

## 4. Решение: Множители Лагранжа

### Первая главная компонента $\mathbf{w}_1$:
$$ L(\mathbf{w}, \lambda) = \mathbf{w}^T C \mathbf{w} - \lambda (\mathbf{w}^T \mathbf{w} - 1) $$

$$ \nabla_{\mathbf{w}} L = 2 C \mathbf{w} - 2 \lambda \mathbf{w} = 0 \implies C \mathbf{w} = \lambda \mathbf{w} $$

$$ \text{Var}(\mathbf{w}) = \mathbf{w}^T C \mathbf{w} = \mathbf{w}^T (\lambda \mathbf{w}) = \lambda (\mathbf{w}^T \mathbf{w}) = \lambda \cdot 1 = \lambda $$

$$ \max \text{Var}(\mathbf{w}_1) \implies \lambda = \lambda_{\max} $$

### Вторая главная компонента $\mathbf{w}_2$:
Дополнительное условие ортогональности: $\mathbf{w}_2^T \mathbf{w}_1 = 0$

$$ L(\mathbf{w}_2, \lambda_2, \phi) = \mathbf{w}_2^T C \mathbf{w}_2 - \lambda_2 (\mathbf{w}_2^T \mathbf{w}_2 - 1) - \phi (\mathbf{w}_2^T \mathbf{w}_1) $$

$$ \nabla_{\mathbf{w}_2} L = 2 C \mathbf{w}_2 - 2 \lambda_2 \mathbf{w}_2 - \phi \mathbf{w}_1 = 0 $$

Умножаем слева на $\mathbf{w}_1^T$:

$$ 2 \mathbf{w}_1^T C \mathbf{w}_2 - 2 \lambda_2 \mathbf{w}_1^T \mathbf{w}_2 - \phi \mathbf{w}_1^T \mathbf{w}_1 = 0 $$

$$ 2 (C \mathbf{w}_1)^T \mathbf{w}_2 - 0 - \phi \cdot 1 = 0 $$

$$ 2 (\lambda_1 \mathbf{w}_1)^T \mathbf{w}_2 = \phi \implies 2 \lambda_1 (\mathbf{w}_1^T \mathbf{w}_2) = \phi \implies 0 = \phi $$

Остается градиент без $\phi$:

$$ 2 C \mathbf{w}_2 - 2 \lambda_2 \mathbf{w}_2 = 0 \implies C \mathbf{w}_2 = \lambda_2 \mathbf{w}_2 $$

---

## 5. Доказательство глобального максимума дисперсии для собственных осей
Базис из собственных векторов $C$: $\{ \mathbf{w}_1, \dots, \mathbf{w}_m \}, \quad \mathbf{w}_i^T \mathbf{w}_j = \delta_{ij}, \quad C \mathbf{w}_i = \lambda_i \mathbf{w}_i$

Произвольный единичный вектор $\mathbf{v}$: 

$$ \mathbf{v} = \sum_{i=1}^m c_i \mathbf{w}_i, \quad \|\mathbf{v}\|^2 = \sum_{i=1}^m c_i^2 = 1 $$

Дисперсия проекции на $\mathbf{v}$:

$$ \text{Var}(\mathbf{v}) = \mathbf{v}^T C \mathbf{v} = \left(\sum_{i=1}^m c_i \mathbf{w}_i^T\right) C \left(\sum_{j=1}^m c_j \mathbf{w}_j\right) = \left(\sum_{i=1}^m c_i \mathbf{w}_i^T\right) \left(\sum_{j=1}^m c_j \lambda_j \mathbf{w}_j\right) $$

$$ \text{Var}(\mathbf{v}) = \sum_{i=1}^m \sum_{j=1}^m c_i c_j \lambda_j (\mathbf{w}_i^T \mathbf{w}_j) \\ $$
Поскольку $\mathbf{w}_i^T \mathbf{w}_j = 0$ при $i \neq j$:

$$ \text{Var}(\mathbf{v}) = \sum_{i=1}^m c_i^2 \lambda_i \quad \text{при} \quad \sum_{i=1}^m c_i^2 = 1 $$

$$ \max_{\{c_i^2\}} \sum_{i=1}^m c_i^2 \lambda_i \implies \{ c_1^2 = 1, \dots, c_m^2 = 0 \} \quad \text{где } \lambda_1 \ge \dots \ge \lambda_m $$

---

## 6. SVD и Вычислительная стабильность PCA
$$ X \in \mathbb{R}^{n \times m} \implies X = U \Sigma V^T $$

Где $U^T U = I_n, \quad V^T V = I_m$. Связь с матрицей ковариации C:

$$ X^T X = (U \Sigma V^T)^T (U \Sigma V^T) = V \Sigma^T U^T U \Sigma V^T = V \Sigma^2 V^T $$

$$ C = \frac{1}{n} X^T X = V \left(\frac{1}{n} \Sigma^2\right) V^T $$

Умножаем справа на $V \quad (V^T V = I)$:

$$ C V = V \left(\frac{1}{n} \Sigma^2\right) \implies C \mathbf{v}_i = \left(\frac{\sigma_i^2}{n}\right) \mathbf{v}_i $$

$$ \mathbf{w}_i = \mathbf{v}_i, \quad \lambda_i = \frac{\sigma_i^2}{n} $$

---

## 7. Численные примеры: Сжатие данных

### Пример 3D $\to$ 2D
Данные $n=4, m=3$ (уже центрированы, $\mu = (0, 0, 0)$):
$$ X = \begin{pmatrix} -4 & -4 & 4 \\ 1 & 2 & -1 \\ -1 & 1 & -4 \\ 4 & 1 & 1 \end{pmatrix}, \quad C = \frac{1}{4} X^T X = \frac{1}{4} \begin{pmatrix} 34 & 21 & -9 \\ 21 & 22 & -21 \\ -9 & -21 & 34 \end{pmatrix} $$

Корни $\det(X^T X - \lambda I) = 0$:

$$ \lambda_1 = 64 \quad \lambda_2 = 25 \quad \lambda_3 = 1 $$

Собственные векторы:

$$ (X^T X - 64 I)\mathbf{w}_1 = \mathbf{0} \implies \mathbf{w}_1 = \begin{pmatrix} 1/\sqrt{3} \\ 1/\sqrt{3} \\ -1/\sqrt{3} \end{pmatrix} $$

$$ (X^T X - 25 I)\mathbf{w}_2 = \mathbf{0} \implies \mathbf{w}_2 = \begin{pmatrix} 1/\sqrt{2} \\ 0 \\ 1/\sqrt{2} \end{pmatrix} $$

Проекция (сжатие):

$$ Z_{3\to 2} = X W = X \begin{pmatrix} \mathbf{w}_1 & \mathbf{w}_2 \end{pmatrix} = \begin{pmatrix} -4\sqrt{3} & 0 \\ 4/\sqrt{3} & 0 \\ 4/\sqrt{3} & -5/\sqrt{2} \\ 4/\sqrt{3} & 5/\sqrt{2} \end{pmatrix} $$

### Пример 2D $\to$ 1D
Данные $n=4, m=2$ (центрированы, $\mu = (0, 0)$):
$$ X = \begin{pmatrix} 2 & 1 \\ -2 & -1 \\ 1 & 2 \\ -1 & -2 \end{pmatrix}, \quad C = \frac{1}{4} X^T X = \frac{1}{4} \begin{pmatrix} 10 & 8 \\ 8 & 10 \end{pmatrix} $$

Корни $\det(X^T X - \lambda I) = 0$:

$$ \lambda_1 = 18 \quad \lambda_2 = 2 $$

Первая ось $w_1$:

$$ (X^T X - 18 I)\mathbf{w}_1 = \mathbf{0} \implies \mathbf{w}_1 = \begin{pmatrix} 1/\sqrt{2} \\ 1/\sqrt{2} \end{pmatrix} $$

Проекция (сжатие на прямую $y=x$):

$$ Z_{2\to 1} = X \mathbf{w}_1 = \begin{pmatrix} 3/\sqrt{2} \\ -3/\sqrt{2} \\ 3/\sqrt{2} \\ -3/\sqrt{2} \end{pmatrix} $$

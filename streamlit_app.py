import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Метод Зейделя", page_icon="🔢", layout="wide")
st.title("Решение системы линейных уравнений методом Зейделя")
st.subheader("Итерационный метод решения СЛАУ")

st.markdown("""
📖 О методе

Метод Зейделя — это итерационный метод решения систем линейных уравнений вида **Ax = b**.
На каждой итерации компоненты вектора обновляются последовательно с использованием уже вычисленных значений:

$$
x_i^{(k+1)} = \\frac{1}{a_{ii}} \\left( b_i - \\sum_{j=1}^{i-1} a_{ij} x_j^{(k+1)} - \\sum_{j=i+1}^{n} a_{ij} x_j^{(k)} \\right)
$$

Точность определяется по формуле:
$$\\varepsilon = \\max_i |x_i^{(k+1)} - x_i^{(k)}|$$
""")

with st.sidebar:
    st.header("Параметры системы")
    
    n = st.number_input("Размерность системы (n)", min_value=2, max_value=5, value=3, step=1)
    
    st.subheader("Матрица A")
    A = []
    for i in range(int(n)):
        row = []
        cols = st.columns(int(n))
        for j in range(int(n)):
            val = cols[j].number_input(f"A[{i+1}][{j+1}]", value=1.0 if i == j else 0.0, key=f"A_{i}_{j}", format="%.2f")
            row.append(val)
        A.append(row)
    
    st.subheader("Вектор b")
    b = []
    for i in range(int(n)):
        val = st.number_input(f"b[{i+1}]", value=1.0, key=f"b_{i}", format="%.2f")
        b.append(val)
    
    st.divider()
    st.header("Параметры точности")
    
    epsilon = st.radio(
        "Выберите точность ε:",
        [0.1, 0.01, 0.001],
        format_func=lambda x: f"ε = {x}"
    )
    
    max_iter = st.number_input("Максимум итераций", min_value=10, max_value=200, value=100, step=10)

tab1, tab2 = st.tabs(["📝 Решение системы", "✅ Проверка диагонального преобладания"])

def seidel_method(A, b, epsilon, max_iter):
    """Метод Зейделя. Возвращает решение, список погрешностей,
    список всех итераций (включая начальное приближение) и сообщение."""
    n = len(b)
    x = np.zeros(n)
    iterations_x = [x.copy()]
    errors = []
    
    for k in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            sum1 = sum(A[i][j] * x_new[j] for j in range(i))
            sum2 = sum(A[i][j] * x[j] for j in range(i+1, n))
            if abs(A[i][i]) < 1e-12:
                return None, None, None, f"Ошибка: нулевой диагональный элемент в строке {i+1}"
            x_new[i] = (b[i] - sum1 - sum2) / A[i][i]
        
        error = max(abs(x_new[i] - x[i]) for i in range(n))
        errors.append(error)
        iterations_x.append(x_new.copy())
        x = x_new
        
        if error < epsilon:
            return x, errors, iterations_x, f"Сошлось за {k+1} итераций"
    
    return x, errors, iterations_x, f"⚠️ Не сошлось за {max_iter} итераций. Погрешность: {errors[-1]:.6f}"

def check_diagonal_dominance(A):
    n = len(A)
    results = []
    has_dominance = True
    for i in range(n):
        diag = abs(A[i][i])
        sum_row = sum(abs(A[i][j]) for j in range(n) if j != i)
        is_dominant = diag > sum_row
        if not is_dominant:
            has_dominance = False
        results.append({
            "Строка": i+1,
            "|a_ii|": f"{diag:.3f}",
            "Σ|a_ij| (j≠i)": f"{sum_row:.3f}",
            "Преобладание": "Да" if is_dominant else "Нет"
        })
    return has_dominance, pd.DataFrame(results)

def transform_to_diagonal_dominance(A, b):
    n = len(A)
    A_transformed = [row[:] for row in A]
    b_transformed = b[:]
    for i in range(n):
        max_row = i
        max_diag = abs(A_transformed[i][i])
        for j in range(i, n):
            if abs(A_transformed[j][i]) > max_diag:
                max_diag = abs(A_transformed[j][i])
                max_row = j
        if max_row != i:
            A_transformed[i], A_transformed[max_row] = A_transformed[max_row], A_transformed[i]
            b_transformed[i], b_transformed[max_row] = b_transformed[max_row], b_transformed[i]
    return A_transformed, b_transformed

def display_system(A, b, title="Система уравнений"):
    st.subheader(f"{title}")
    n = len(A)
    for i in range(n):
        eq = ""
        for j in range(n):
            coef = A[i][j]
            sign = "+" if coef >= 0 and j > 0 else ""
            if abs(coef) > 1e-10:
                if j == 0:
                    eq += f"{coef:.3f}·x_{j+1}"
                else:
                    eq += f"{sign}{coef:.3f}·x_{j+1}"
        eq += f" = {b[i]:.3f}"
        st.latex(eq)

def display_iterations(iterations_x, errors, A, b, epsilon):
    """Показывает таблицу итераций и пояснения к первой итерации."""
    n = len(b)
    st.subheader("Ход решения (итерации)")
    
    with st.expander("🔍 Как вычисляются значения на каждой итерации?", expanded=False):
        st.markdown(rf"**Формула метода Зейделя для системы $n={n}$:**")
        # Общая формула
        st.latex(r"x_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j=1}^{i-1} a_{ij} x_j^{(k+1)} - \sum_{j=i+1}^{n} a_{ij} x_j^{(k)} \right)")
        
        if n <= 3:
            st.markdown("**В раскрытом виде:**")
            for i in range(n):
                terms_old = " + ".join([f"{A[i][j]:.3f}·x_{j+1}^{{(k)}}" for j in range(i+1, n) if abs(A[i][j])>1e-10])
                terms_new = " + ".join([f"{A[i][j]:.3f}·x_{j+1}^{{(k+1)}}" for j in range(i) if abs(A[i][j])>1e-10])
                if not terms_new:
                    terms_new = "0"
                if not terms_old:
                    terms_old = "0"
                st.latex(rf"x_{i+1}^{{(k+1)}} = \frac{{1}}{{{A[i][i]:.3f}}} \left( {b[i]:.3f} - ({terms_new}) - ({terms_old}) \right)")
        
        if len(iterations_x) >= 2:
            x_old = iterations_x[0]
            x_new = iterations_x[1]
            st.markdown("**Пример для первой итерации (начальное приближение – нулевой вектор):**")
            for i in range(n):
                terms_new = []
                terms_old = []
                for j in range(n):
                    if j != i:
                        coeff = A[i][j]
                        if abs(coeff) > 1e-12:
                            if j < i:
                                terms_new.append(f"{coeff:.3f}·{x_new[j]:.6f}")
                            else:
                                terms_old.append(f"{coeff:.3f}·{x_old[j]:.6f}")
                sum_new = " + ".join(terms_new) if terms_new else "0"
                sum_old = " + ".join(terms_old) if terms_old else "0"
                st.latex(rf"x_{i+1}^{{(1)}} = \frac{{1}}{{{A[i][i]:.3f}}} \left( {b[i]:.3f} - ({sum_new}) - ({sum_old}) \right) = {x_new[i]:.6f}")
    
    iter_data = []
    for idx, x_vals in enumerate(iterations_x):
        row = {"Итерация": idx}
        for i in range(n):
            row[f"x_{i+1}"] = f"{x_vals[i]:.6f}"
        if idx > 0:
            row["Погрешность ε"] = f"{errors[idx-1]:.6e}"
        else:
            row["Погрешность ε"] = "—"
        iter_data.append(row)
    
    df_iter = pd.DataFrame(iter_data)
    
    max_display_rows = 20
    if len(df_iter) > max_display_rows:
        st.info(f"Выполнено {len(df_iter)-1} итераций. Показаны первые 10 и последние 5.")
        df_top = df_iter.head(10)
        df_bottom = df_iter.tail(5)
        sep_row = {col: "..." for col in df_iter.columns}
        sep_row["Итерация"] = "..."
        df_sep = pd.DataFrame([sep_row])
        df_display = pd.concat([df_top, df_sep, df_bottom], ignore_index=True)
    else:
        df_display = df_iter
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    if len(errors) > 0:
        st.caption(f"**Достигнутая точность:** {errors[-1]:.6e} (требовалось {epsilon})")

def solve_with_epsilon(A, b, eps, max_iter):
    solution, _, _, _ = seidel_method(A, b, eps, max_iter)
    return solution

with tab1:
    display_system(A, b, "Введенная система уравнений")
    
    if st.button("Решить систему", type="primary", use_container_width=True):
        A_np = np.array(A)
        b_np = np.array(b)
        
        if abs(np.linalg.det(A_np)) < 1e-10:
            st.error("Матрица вырожденная! Система не имеет единственного решения.")
        else:
            with st.spinner(f"Решаем с точностью ε = {epsilon}..."):
                solution, errors, all_x, message = seidel_method(A, b_np, epsilon, max_iter)
            
            if solution is not None:
                st.success(message)
                
                display_iterations(all_x, errors, A, b, epsilon)
                
                st.subheader("Определение точности")
                st.latex(r"\varepsilon = \max_i |x_i^{(k+1)} - x_i^{(k)}|")
                st.write(f"**Заданная точность:** ε = {epsilon}")
                if errors:
                    last_error = errors[-1]
                    st.info(f"**Последняя достигнутая погрешность:** {last_error:.8f}")
                    if last_error < epsilon:
                        st.success(f"{last_error:.8f} < {epsilon} → процесс остановлен")
                    else:
                        st.warning(f"{last_error:.8f} > {epsilon} → требуются дополнительные итерации")
                
                st.subheader(f"ОТВЕТ (при ε = {epsilon}):")
                answer_cols = st.columns(int(n))
                for i in range(int(n)):
                    with answer_cols[i]:
                        st.metric(f"x_{i+1} =", f"{solution[i]:.3f}")
                
                vars_str = ', '.join([f"x_{i+1}" for i in range(n)])
                vals_str = ', '.join([f"{solution[i]:.3f}" for i in range(n)])
                st.success(f"""
**Решение системы линейных уравнений с точностью ε = {epsilon}:**

$({vars_str}) = ({vals_str})$

**Погрешность:** {last_error:.2e} {'<' if last_error < epsilon else '>'} {epsilon}
""")
                
                st.subheader("Сравнение решений при разных ε")
                sol_01 = solve_with_epsilon(A, b_np, 0.1, max_iter)
                sol_001 = solve_with_epsilon(A, b_np, 0.01, max_iter)
                sol_0001 = solve_with_epsilon(A, b_np, 0.001, max_iter)
                
                data = []
                for eps_val, sol, name in [(0.1, sol_01, "ε = 0.1"), (0.01, sol_001, "ε = 0.01"), (0.001, sol_0001, "ε = 0.001")]:
                    if sol is not None and len(sol) == n:
                        row_dict = {"Точность (ε)": name}
                        for i in range(n):
                            row_dict[f"x_{i+1}"] = f"{sol[i]:.6f}"
                        data.append(row_dict)
                if data:
                    df_results = pd.DataFrame(data)
                    st.dataframe(df_results, use_container_width=True, hide_index=True)
                
                if errors:
                    st.subheader("График сходимости метода")
                    chart_data = pd.DataFrame({"Погрешность": errors[:50] if len(errors) > 50 else errors})
                    st.line_chart(chart_data, y_label="Погрешность", x_label="Номер итерации")
                    st.caption(f"Финальная погрешность: {errors[-1]:.2e}")

with tab2:
    st.subheader("Проверка диагонального преобладания")
    st.markdown("""
    ### Условие сходимости метода Зейделя:
    Для сходимости метода необходимо **диагональное преобладание**:
    $$|a_{ii}| > \\sum_{j=1, j \\neq i}^{n} |a_{ij}|, \\quad i = 1, 2, ..., n$$
    То есть модуль диагонального элемента должен быть строго больше суммы модулей всех остальных элементов в строке.
    """)
    
    has_dominance, df_dominance = check_diagonal_dominance(A)
    st.subheader("Анализ матрицы A:")
    st.dataframe(df_dominance, use_container_width=True, hide_index=True)
    
    if has_dominance:
        st.success("Условие диагонального преобладания ВЫПОЛНЕНО! Метод Зейделя будет сходиться.")
    else:
        st.warning("Условие диагонального преобладания НЕ ВЫПОЛНЕНО! Метод может не сойтись или сходиться медленно.")
        st.subheader("Преобразование системы к виду с диагональным преобладанием")
        st.markdown("**Метод преобразования:** перестановка строк для максимизации диагональных элементов.")
        A_transformed, b_transformed = transform_to_diagonal_dominance(A, b)
        has_dominance_transformed, df_dominance_transformed = check_diagonal_dominance(A_transformed)
        display_system(A_transformed, b_transformed, "Преобразованная система (после перестановки строк)")
        st.subheader("Анализ преобразованной матрицы:")
        st.dataframe(df_dominance_transformed, use_container_width=True, hide_index=True)
        
        if has_dominance_transformed:
            st.success("После преобразования диагональное преобладание ДОСТИГНУТО! Рекомендуется использовать преобразованную систему.")
            if st.button("Использовать преобразованную систему для решения", use_container_width=True):
                st.session_state.A_transformed = A_transformed
                st.session_state.b_transformed = b_transformed
                st.session_state.use_transformed = True
                st.success("Преобразованная система загружена! Перейдите на вкладку 'Решение системы' и нажмите 'Решить систему'.")
                st.info("Совет: обновите матрицу A и вектор b в боковой панели, используя значения выше.")
                st.write("**Матрица A (преобразованная):**")
                for i in range(len(A_transformed)):
                    row_str = "[" + ", ".join([f"{A_transformed[i][j]:.3f}" for j in range(len(A_transformed))]) + "]"
                    st.code(f"Строка {i+1}: {row_str}")
                st.write("**Вектор b (преобразованный):**")
                st.code(f"[{', '.join([f'{b_transformed[i]:.3f}' for i in range(len(b_transformed))])}]")
        else:
            st.error("Даже после перестановки строк диагональное преобладание не достигнуто. Рекомендуется использовать другие методы (Гаусса, Крамера).")
    
    st.subheader("Визуализация диагонального преобладания (исходная система)")
    n_matrix = int(n)
    comparison_data = []
    for i in range(n_matrix):
        diag = abs(A[i][i])
        sum_others = sum(abs(A[i][j]) for j in range(n_matrix) if j != i)
        comparison_data.append({
            "Строка": i+1,
            "|a_ii|": diag,
            "Σ|a_ij| (j≠i)": sum_others,
            "Разница": diag - sum_others
        })
    df_compare = pd.DataFrame(comparison_data)
    st.bar_chart(df_compare.set_index("Строка")[["|a_ii|", "Σ|a_ij| (j≠i)"]])
    st.caption("Для сходимости синий столбец (|a_ii|) должен быть выше голубого (Σ|a_ij|)")

# Session 1 In-Class Quiz: Fundamentals of Pandas

Based on `BigData-session-1.ipynb`. 10 questions: true/false and multiple choice.
Answer without running code first, then check your answers in the notebook.

Unless stated otherwise, assume:

```python
import pandas
MEM_USAGE = pandas.Series([0.16, 0.07, 0.23, 0.24, 0.14],
                          index=[3, 1, 4, 2, 7])
df_mystery = pandas.read_csv("sherlock/sherlock_mystery.csv")   # 200 rows, 14 columns
df6 = df_mystery.head(6).copy()                                  # row labels 0..5
```

---

### Q1. True or False

When you create a Series with `pandas.Series([0.16, 0.07, 0.23])` and no `index=` argument, its labels start at `1`.

- [ ] True
- [ ] False

---

### Q2. Multiple choice

What does `MEM_USAGE[3]` return?

- A. `0.24` (the 4th element)
- B. `0.16`
- C. `0.23`
- D. `KeyError`

---

### Q3. True or False

After running the code below, `MEM_USAGE[3]` is still `0.16`.

```python
COPY_USAGE = MEM_USAGE
COPY_USAGE[3] = 1.23
```

- [ ] True
- [ ] False

---

### Q4. Multiple choice

`avatars` has 5 rows and the columns `Name`, `Weight`, `Height`. What does `avatars.shape` return?

- A. `15`
- B. `(3, 5)`
- C. `(5, 3)`
- D. `5`

---

### Q5. Multiple choice

`df_mystery` has 14 columns, and `ApplicationName` holds text (`Facebook`, `WhatsApp`). How many columns appear in `df_mystery.describe()`?

- A. 14
- B. 13
- C. 8
- D. 1

---

### Q6. Multiple choice

Which pair of object types do `df6['ApplicationName']` and `df6[['ApplicationName']]` return, in that order?

- A. `Series`, `Series`
- B. `DataFrame`, `DataFrame`
- C. `Series`, `DataFrame`
- D. `DataFrame`, `Series`

---

### Q7. True or False

`df6.loc[2:5]` returns the rows labeled 2, 3, 4 **and 5**.

- [ ] True
- [ ] False

---

### Q8. Multiple choice

Which row labels does `df6[[True, False, False, True, True, False]]` return?

- A. `0, 3, 4`
- B. `1, 2, 5`
- C. `0, 1, 2`
- D. It selects columns, not rows

---

### Q9. Multiple choice

Which statement sets `CPU_USAGE` to `0.01` in the row labeled `2` of `df6`?

- A. `df6[2, 'CPU_USAGE'] = 0.01`
- B. `df6.loc[2, 'CPU_USAGE'] = 0.01`
- C. `df6.loc['CPU_USAGE', 2] = 0.01`
- D. `df6.CPU_USAGE.loc = 0.01`

---

### Q10. True or False

In a box-and-whisker plot such as `df_mystery['vsize'].plot(kind='box')`, the line drawn inside the box marks the **mean**.

- [ ] True
- [ ] False

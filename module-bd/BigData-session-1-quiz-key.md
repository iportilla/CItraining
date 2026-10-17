# Session 1 In-Class Quiz: Solution Key

Answers to `BigData-session-1-quiz.md`, checked against `sherlock/sherlock_mystery.csv`.

| Q | Answer | Notebook section |
|---|--------|------------------|
| 1 | False | 2.1 Creating a Series |
| 2 | B | 2.2.1 Accessing a single element |
| 3 | False | 2.4 Creating a copy |
| 4 | C | 3.1.2 How big is my data? |
| 5 | B | 3.2.1 Initial DataFrame exploration |
| 6 | C | 3.3.1 / 3.4 Exercise 1 |
| 7 | True | 3.3.6 Slicing operator |
| 8 | A | 3.3.3 Filtering rows by Boolean |
| 9 | B | 3.3.5 Updating a cell |
| 10 | False | 4.2 Box-and-whisker plot |

---

**Q1. False.** The default index is an integer sequence starting at `0`, following Python's convention for numbering array elements.

**Q2. B, `0.16`.** `[]` on a Series uses the **label**, not the position. Label `3` is the first element, `0.16`. Position 3 (option A) would be label `2`, which holds `0.24`.

**Q3. False.** `COPY_USAGE = MEM_USAGE` does not copy anything. Both names refer to the same object, so the change shows through both, and `MEM_USAGE[3]` becomes `1.23`. Use `MEM_USAGE.copy()` to get an independent copy.

**Q4. C, `(5, 3)`.** `shape` is a tuple of (rows, columns). `len(avatars)` gives `5` (option D), the number of rows only.

**Q5. B, 13.** `describe()` keeps only numerical columns and quietly drops the text column `ApplicationName`, so 14 − 1 = 13. It has 8 **rows** (count, mean, std, min, 25%, 50%, 75%, max), which is where option C comes from.

**Q6. C.** A single column name gives a `Series` (one dimension). A **list** of column names, even a list of one, gives a `DataFrame`. The same rule applies to rows: `df6.loc[1]` is a Series, `df6.loc[[1]]` is a DataFrame.

**Q7. True.** Slices in `.loc[]` use labels and include **both** endpoints, so the result is rows `2, 3, 4, 5`. Plain Python slices exclude the end, so this one surprises people.

**Q8. A, `0, 3, 4`.** A Boolean list in `[]` selects **rows**, keeping those at the `True` positions (1st, 4th, 5th). This is the one case where `[]` on a DataFrame picks rows instead of columns.

**Q9. B.** `.loc[row, column]` reads or writes a cell, with the row label first and the column second. Option A is a trap: `[]` treats `(2, 'CPU_USAGE')` as one column name, so it quietly **adds a new column** called `(2, 'CPU_USAGE')` and leaves the real cell at `0.23`. Option C has the order reversed. The notebook says to use `.loc[]` for updates and to treat the result of `[]` as read-only.

**Q10. False.** The line inside the box is the **median** (50%). The box edges are the 25% and 75% quartiles, and the whiskers reach toward min and max. The mean is not drawn by default. For `vsize` the median is about 2.11e9 and the mean about 2.18e9, so the two are not the same.

---

### Optional bonus question (from the notebook)

What is the mean of `num_threads`? Answer: **71.365**. The median is 62.0, a likely wrong pick.

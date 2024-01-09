from optimization_algorithm import comparison
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'comparison' is a dictionary containing your data
data = comparison

# Create a PrettyTable
table = PrettyTable()
table.field_names = [
    "S/N",
    "Benchmark Function",
    "mpso",
    "pso",
    "T test",
    "Better Algorithm",
]

count = 0
for entry in data.values():
    count += 1
    benchmark_function = entry["benchmarkFunction"].value
    mpso_values = entry["mpso_values"]
    pso_values = entry["pso_values"]
    t_test = entry["t_test"]
    better_algorithm = entry["better_algorithm"]

    mpso_column = f"Mean: {mpso_values['mean']}\nStd Dev: {mpso_values['standard_deviation']}\nIter Size: {mpso_values['iteration_count']}"
    pso_column = f"Mean: {pso_values['mean']}\nStd Dev: {pso_values['standard_deviation']}\nIter Size: {pso_values['iteration_count']}"

    table.add_row(
        [count, benchmark_function, mpso_column, pso_column, t_test, better_algorithm]
    )

fig, ax = plt.subplots(figsize=(10, 8))
ax.axis("off")
table_data = [table.field_names] + table._rows
table_ax = ax.table(
    cellText=table_data,
    cellLoc="center",
    loc="center",
)
table_ax.auto_set_font_size(False)
table_ax.set_fontsize(8)

table_ax.scale(1, 3)
table_ax.auto_set_column_width([i for i in range(len(table.field_names))])

plt.savefig("table_visualization.png", bbox_inches="tight", pad_inches=0.5)


# Convert the PrettyTable to a Pandas DataFrame
df = pd.DataFrame(table._rows[1:], columns=table.field_names)

# Export the DataFrame to an Excel file
df.to_csv(
    "table_data.csv",
    index=False,
)

plt.show()

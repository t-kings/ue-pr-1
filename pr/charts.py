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
    "PSO",
    "MPSO",
    "MPSO-1",
    "MPSO-2",
]

count = 0
for entry in data.values():
    count += 1
    benchmark_function = entry["benchmarkFunction"].value
    mpso_values = entry["mpso_values"]
    pso_values = entry["pso_values"]
    mpso1_values = entry["mpso1_values"]
    mpso2_values = entry["mpso2_values"]

    pso_column = f"Mean: {pso_values['mean']}\nStd Dev: {pso_values['standard_deviation']}\nT test: {pso_values['t_test']}\nIteration Size: {pso_values['iteration_count']}\nRank: {pso_values['rank']}"
    mpso_column = f"Mean: {mpso_values['mean']}\nStd Dev: {mpso_values['standard_deviation']}\nT test: \nIteration Size: {mpso_values['iteration_count']}\nRank: {mpso_values['rank']}"
    mpso1_column = f"Mean: {mpso1_values['mean']}\nStd Dev: {mpso1_values['standard_deviation']}\nT test: {mpso1_values['t_test']}\nIteration Size: {mpso1_values['iteration_count']}\nRank: {mpso1_values['rank']}"
    mpso2_column = f"Mean: {mpso2_values['mean']}\nStd Dev: {mpso2_values['standard_deviation']}\nT test: {mpso2_values['t_test']}\nIteration Size: {mpso2_values['iteration_count']}\nRank: {mpso2_values['rank']}"

    table.add_row(
        [count, benchmark_function, pso_column, mpso_column, mpso1_column, mpso2_column]
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

table_ax.scale(1, 4)
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

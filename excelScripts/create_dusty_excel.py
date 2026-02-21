import pandas as pd

# Sheet 1
sheet1 = pd.DataFrame({
    'MAXDIM': [25, 50, 75, 100, 125, 150],
    'dusty.f_O0': [''] * 6,
    'dusty.f_O1': [''] * 6,
    'dusty.f_O2': [''] * 6,
    'dusty.f_O3': [''] * 6,
    'dustycpp_O0': [''] * 6,
    'dustycpp_O1': [''] * 6,
    'dustycpp_O2': [''] * 6,
    'dustycpp_O3': [''] * 6,
    'dustyf90_O0': [''] * 6,
    'dustyf90_O1': [''] * 6,
    'dustyf90_O2': [''] * 6,
    'dustyf90_O3': [''] * 6,
})

# Sheet 2
sheet2 = pd.DataFrame({
    'MAXDIM': [25, 50, 75, 100, 125, 150],
    'CPP_Baseline': ['', '', '', 33.4, '', ''],
    'CPP_OPT0': ['', '', '', 32.6, '', ''],
    'CPP_OPT1': ['', '', '', 32.4, '', ''],
    'CPP_OPT2': ['', '', '', 29.5, '', ''],
    'CPP_OPT3': ['', '', '', 25.5, '', ''],
    'CPP_OPT4': ['', '', '', 21.9, '', ''],
    'F90_Baseline': [''] * 6,
    'F90_OPT1': [''] * 6,
    'F90_OPT2': [''] * 6,
    'F90_OPT3': [''] * 6,
    'F90_OPT4': [''] * 6,
})

# Sheet 3
sheet3 = pd.DataFrame({
    'MAXDIM': [25, 50, 75, 100, 125, 150],
    'CPP_Tuned_O0': ['', '', '', 21.9, '', ''],
    'CPP_Tuned_O1': [''] * 6,
    'CPP_Tuned_O2': [''] * 6,
    'CPP_Tuned_O3': [''] * 6,
    'F90_Tuned_O0': [''] * 6,
    'F90_Tuned_O1': [''] * 6,
    'F90_Tuned_O2': [''] * 6,
    'F90_Tuned_O3': [''] * 6,
})

# Sheet 4
sheet4 = pd.DataFrame({
    'Optimization': [
        'OPT0: Skip Renorm',
        'OPT1: Loop Interchange', 
        'OPT2: Switch to If-Else',
        'OPT3: Partial Loop Unroll',
        'OPT4: Full Loop Unroll',
        'TOTAL'
    ],
    'Description': [
        'Eliminate redundant normalization',
        'Better cache locality for CM matrix',
        'Replace switch with if-else chain',
        'Reduce modulo operations (with r+1 mod 4)',
        'Eliminate all modulo in main loop',
        'Combined effect'
    ],
    'CPP_Time_Before': [33.4, 32.6, 32.4, 29.5, 25.5, 33.4],
    'CPP_Time_After': [32.6, 32.4, 29.5, 25.5, 21.9, 21.9],
    'CPP_Speedup': ['', '', '', '', '', ''],
    'F90_Time_Before': [''] * 6,
    'F90_Time_After': [''] * 6,
    'F90_Speedup': [''] * 6,
})

# Write to Excel
with pd.ExcelWriter('optimization_data.xlsx', engine='openpyxl') as writer:
    sheet1.to_excel(writer, sheet_name='Baseline_Compiler_Opts', index=False)
    sheet2.to_excel(writer, sheet_name='Code_Optimizations', index=False)
    sheet3.to_excel(writer, sheet_name='Final_Tuned', index=False)
    sheet4.to_excel(writer, sheet_name='Individual_Impact', index=False)

print("Excel file created: optimization_data.xlsx")

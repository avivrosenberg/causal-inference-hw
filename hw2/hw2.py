import pandas as pd
import numpy as np
import itertools as it


factors = {
    'Z': [],
    'X': ['Z'],
    'T': ['Z','X'],
    'Y': ['T','Z','X'],
    'W': ['X','Y'],
}

df = pd.read_csv('hw2.csv')

# All variables are binary
vals_range = range(2)

for main_var, cond_vars in factors.items():

    cond_var_value_ranges = [vals_range]*len(cond_vars)
    cond_var_values = it.product(*cond_var_value_ranges)

    for cond_values in cond_var_values:

        # Build an index selecting rows where cond_vars are equal to the current
        # cond_values tuple
        idx = np.full(len(df), True)
        cond_strs = []
        for i, val in enumerate(cond_values):
            idx &= df[cond_vars[i]] == val
            cond_strs.append(f'{cond_vars[i]}={val}')

        cond_rows = df[idx]
        den = len(cond_rows)

        # Filter on the values of the main variable
        for main_val in vals_range:
            en = np.sum(cond_rows[main_var] == main_val)

            # Print the proba
            print(f'p({main_var}={main_val}|{str.join(",", cond_strs)}) = {en}/{den}')

    print('')

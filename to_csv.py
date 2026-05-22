import pandas as pd
file = pd.read_csv("hyperopt_monthly_regression_walk_forward_summary.csv")
file['stability_adjusted_hitrate'] = file['mean_hit_rate']-file['std_hit_rate']
file.to_csv("Nieuw.csv")
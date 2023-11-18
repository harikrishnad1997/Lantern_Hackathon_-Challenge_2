import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, f_classif, chi2, SelectKBest

class FeatureSelectionPipeline:
    def __init__(self, df, rna_columns, mut_columns, target_column):
        self.df = df
        self.rna_columns = rna_columns
        self.mut_columns = mut_columns
        self.target_column = target_column

    def create_rna_pipeline(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.feature_selection import VarianceThreshold

        X = self.df[self.rna_columns]

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('variance_threshold', VarianceThreshold(threshold=0.1))
        ])

        X_transformed = pipeline.fit_transform(X)

        selected_features = pipeline.named_steps['variance_threshold'].get_support(indices=True)

        excluded_columns = np.setdiff1d(range(len(X.columns)), selected_features)
        rna_excluded_columns = X.columns[excluded_columns]

        rna_var_columns = list(set(self.rna_columns) - set(rna_excluded_columns))
        return rna_var_columns

    def perform_anova_test(self, selected_columns):
        y = self.df[self.target_column]

        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(self.df[selected_columns], y)

        selected_indices = selector.get_support(indices=True)

        f_values = selector.scores_
        p_values = selector.pvalues_

        anova_test_result_df = pd.DataFrame({
            'Feature': selected_columns,
            'F-Value': f_values,
            'P-Value': p_values
        })

        return anova_test_result_df[anova_test_result_df["P-Value"] < 0.05].sort_values(by="F-Value", ascending=False)

    def perform_chi_square_test(self):
        y = self.df[self.target_column]

        selector = SelectKBest(score_func=chi2, k='all')
        selector.fit(self.df[self.mut_columns], y)

        selected_indices = selector.get_support(indices=True)

        f_values = selector.scores_
        p_values = selector.pvalues_

        chi_test_result_df = pd.DataFrame({
            'Feature': self.mut_columns,
            'F-Value': f_values,
            'P-Value': p_values
        })

        return chi_test_result_df.sort_values(by="F-Value", ascending=False)

    def get_final_columns(self, anova_test_result_df, chi_test_result_df):
        rna_final_columns = list(set(anova_test_result_df[anova_test_result_df["P-Value"] <= 0.05]["Feature"]))

        mut_final_columns = list(set(chi_test_result_df[chi_test_result_df["P-Value"] <= 0.05]["Feature"]))

        final_columns = [self.target_column, 'type']
        final_columns.extend(mut_final_columns)
        final_columns.extend(rna_final_columns)
        final_columns = list(set(final_columns))

        return final_columns

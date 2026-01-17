import numpy as np
from numpy.typing import NDArray


def compensate_background(combined_df):
    from scipy.interpolate import PchipInterpolator
    
    def _baseline_on_key_point(x_minima: NDArray[np.int64],
                                y_minima: NDArray[np.float64],
                                x: NDArray[np.int64]) -> NDArray[np.float64]:
        pchip = PchipInterpolator(x_minima, y_minima)
        baseline = pchip(x)
        return baseline
    
    
    def _direct_method(x_array: NDArray[np.int64],
                        y_array: NDArray[np.float64],
                        num_points: int) -> NDArray[np.int64]:
    
        y_chunks = np.array_split(y_array, num_points)
        x_chunks = np.array_split(x_array, num_points)
    
        min_indices = np.array([
            x_chunk[np.argmin(y_chunk)]
            for y_chunk, x_chunk in zip(y_chunks, x_chunks)
        ])
    
        return min_indices
    
    
    def _fix_points(key_points: NDArray[np.int64],
                    eps: int,
                    y_array: NDArray[np.float64]) -> NDArray[np.float64]:
        starts = np.maximum(key_points - eps, 0)
        ends = np.minimum(key_points + eps, len(y_array))
    
        new_points = np.array([
            start + np.argmin(y_array[start:end])
            for start, end in zip(starts, ends)
        ])
    
        return np.unique(new_points)
 
    for col in combined_df.columns:
        if col == 'Wavenumber':
            continue
        y = combined_df[col]

        # y = self.df["Intensity"]
        # x = self.df["Wavenumber"]

        indexes = [i for i in range(len(y))]
        y_array = np.array(y)
        indexes_array = np.array(indexes)

        y_dim: int = np.shape(y_array)[0]
        eps: int = max(int(0.02 * y_dim), 1)
        best_score_result: int = y_dim
        best_key_points: np.array([], dtype=np.int64)

        for num_points in range(3, 10, 1):
            direct_key_point = _direct_method(indexes_array, y_array, num_points)
            fixed_points_without_zero = _fix_points(direct_key_point, eps, y_array)
            baseline_without_zero = _baseline_on_key_point(
                fixed_points_without_zero,
                y_array[fixed_points_without_zero],
                indexes_array
            )
            score_without_zero = np.sum(y_array < baseline_without_zero)

            points_with_zero = np.unique(np.append(direct_key_point, 0))
            fixed_points_with_zero = _fix_points(points_with_zero, eps, y_array)
            baseline_with_zero = _baseline_on_key_point(
                fixed_points_with_zero,
                y_array[fixed_points_with_zero],
                indexes_array
            )
            score_with_zero = np.sum(y_array < baseline_with_zero)

            if score_without_zero <= score_with_zero:
                current_score = score_without_zero
                current_points = fixed_points_without_zero
            else:
                current_score = score_with_zero
                current_points = fixed_points_with_zero

            if current_score < best_score_result:
                best_score_result = current_score
                best_key_points = current_points

        y_for_best_x = y_array[best_key_points]

        baseline = _baseline_on_key_point(best_key_points, y_for_best_x, indexes_array)

        compensated = y - baseline

        combined_df[col] = compensated - np.min(compensated)

    return combined_df
        # self.table.setModel(PandasModel(combined_df))
        # self.plot_all_combined_spectra()
        # self.plot_canvas.plot(self.df["Wavenumber"], self.df["Compensated"])
        # self.current_spectrum_col = "Compensated"


def normalize_total(combined_df):
    max_value = combined_df.iloc[:, 1:].max(axis=0).max()
    combined_df.iloc[:, 1:] = combined_df.iloc[:, 1:] / max_value
    return combined_df

def normalize_in_range(combined_df, wavelenght_range: tuple[float, float]):
    wavelenghts = combined_df.iloc[:, 0]
    start = wavelenghts[wavelenghts <= float(wavelenght_range[0])].idxmax()
    stop = wavelenghts[wavelenghts <= float(wavelenght_range[1])].idxmax()
    max_value = combined_df.iloc[start:stop, 1:].max(axis=0).max()
    combined_df.iloc[:, 1:] = combined_df.iloc[:, 1:] / max_value
    return combined_df
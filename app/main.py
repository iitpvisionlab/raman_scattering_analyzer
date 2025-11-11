import sys
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QHBoxLayout,
    QToolBar,
    QLineEdit,
    QCheckBox,
    QComboBox,
    QDialog,
    QLabel,
)
from PyQt6.QtCore import QAbstractTableModel, Qt, QSize, QModelIndex
from PyQt6.QtGui import QAction

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class PandasModel(QAbstractTableModel):
    def __init__(self, dataframe: pd.DataFrame):
        super().__init__()
        self._dataframe = dataframe

    def rowCount(self, parent=None):
        return len(self._dataframe)

    def columnCount(self, parent=None):
        return len(self._dataframe.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            value = self._dataframe.iloc[index.row(), index.column()]
            return str(value)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return self._dataframe.columns[section]
            else:
                return str(self._dataframe.index[section])
        return None


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(5, 3))
        self.axes = fig.add_subplot(111)
        super().__init__(fig)

    def plot(self, x, y):
        self.axes.clear()
        self.axes.plot(x, y)
        self.axes.set_xlabel("Wavenumber")
        self.axes.set_ylabel("Intensity")
        self.draw()


class ExportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Spectrum")
        self.setFixedSize(350, 180)
        layout = QVBoxLayout(self)

        # File name entry with file dialog button
        file_layout = QHBoxLayout()
        self.file_edit = QLineEdit(self)
        self.file_edit.setPlaceholderText("Choose filename for spectrum")
        file_btn = QPushButton("Browse...", self)
        file_btn.clicked.connect(self.open_file_dialog)
        file_layout.addWidget(QLabel("Spectrum file:", self))
        file_layout.addWidget(self.file_edit)
        file_layout.addWidget(file_btn)
        layout.addLayout(file_layout)

        # Option to save image
        img_layout = QHBoxLayout()
        self.save_img_checkbox = QCheckBox("Save plot image", self)
        self.save_img_checkbox.setChecked(True)
        img_layout.addWidget(self.save_img_checkbox)
        self.img_format_combo = QComboBox(self)
        self.img_format_combo.addItems(["PNG", "TIFF"])
        img_layout.addWidget(QLabel("Image format:", self))
        img_layout.addWidget(self.img_format_combo)
        layout.addLayout(img_layout)

        # Ok/Cancel buttons
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("Export", self)
        cancel_btn = QPushButton("Cancel", self)
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def open_file_dialog(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Choose Save File", "", "Text Files (*.dpt);;All Files (*)"
        )
        if filename:
            self.file_edit.setText(filename)

    def get_options(self):
        return {
            "filename": self.file_edit.text(),
            "save_image": self.save_img_checkbox.isChecked(),
            "image_format": self.img_format_combo.currentText(),
        }


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectrum Loader")
        self.setGeometry(100, 100, 800, 600)
        widget = QWidget()

        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, toolbar)

        load_action = QAction("Load spectrum", self)
        load_action.triggered.connect(self.load_spectrum)
        toolbar.addAction(load_action)

        export_action = QAction("Export spectrum", self)
        export_action.triggered.connect(self.export_spectrum)
        toolbar.addAction(export_action)

        compensate_action = QAction("Compensate background", self)
        compensate_action.triggered.connect(self.compensate_background)
        toolbar.addAction(compensate_action)

        # Layout for table and plot (side by side)
        table_plot_layout = QHBoxLayout()
        self.table = QTableView()
        table_plot_layout.addWidget(self.table)
        self.plot_canvas = PlotCanvas(self)
        table_plot_layout.addWidget(self.plot_canvas)

        # Main layout with button on top
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.addLayout(table_plot_layout)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.df = None
        self.current_spectrum_col = None

        self.table.doubleClicked.connect(self.plot_column)

    def plot_column(self, index: QModelIndex):
        if self.df is not None:
            col = index.column()
            col_name = self.df.columns[col]
            # Skip plotting wavenumber column (only intensity columns)
            if col_name.lower() != "wavenumber":
                x = self.df["Wavenumber"]
                y = self.df[col_name]
                self.plot_canvas.plot(x, y)
                self.statusBar().showMessage(f"Plotting column: {col_name}")
                self.current_spectrum_col = col_name

    def load_spectrum(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Spectrum File", "", "Text Files (*.dpt);;All Files (*)"
        )
        if file_name:
            self.df = pd.read_csv(
                file_name, sep="\t", header=None, names=["Wavenumber", "Intensity"]
            )
            self.table.setModel(PandasModel(self.df))
            self.plot_canvas.plot(self.df["Wavenumber"], self.df["Intensity"])
            self.current_spectrum_col = "Intensity"

    def export_spectrum(self):
        if self.df is None or not hasattr(self, "current_spectrum_col"):
            return  # Nothing to export
        col_name = self.current_spectrum_col
        if col_name is None or col_name.lower() == "wavenumber":
            return  # No valid spectrum selected
        dialog = ExportDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            opts = dialog.get_options()
            export_df = self.df[["Wavenumber", col_name]].copy()
            if not opts["filename"].endswith(".dpt"):
                opts["filename"] = opts["filename"] + ".dpt"
            export_df.to_csv(opts["filename"], sep="\t", index=False, header=False)
            if opts["save_image"]:
                img_ext = opts["image_format"].lower()
                img_file = opts["filename"].rsplit(".", 1)[0] + "." + img_ext
                self.plot_canvas.figure.savefig(img_file, format=img_ext)

    def compensate_background(self):
        from scipy.interpolate import PchipInterpolator
        import numpy as np
        from numpy.typing import NDArray

        # Placeholder for compensation logic
        def baseline_on_key_point(x_minima: NDArray[np.int64],
                                  y_minima: NDArray[np.float64],
                                  x: NDArray[np.int64]) -> NDArray[np.float64]:
            pchip = PchipInterpolator(x_minima, y_minima)
            baseline = pchip(x)
            return baseline
        
        
        def direct_method(x_array: NDArray[np.int64],
                          y_array: NDArray[np.float64],
                          num_points: int) -> NDArray[np.int64]:
        
            y_chunks = np.array_split(y_array, num_points)
            x_chunks = np.array_split(x_array, num_points)
        
            min_indices = np.array([
                x_chunk[np.argmin(y_chunk)]
                for y_chunk, x_chunk in zip(y_chunks, x_chunks)
            ])
        
            return min_indices
        
        
        def fix_points(key_points: NDArray[np.int64],
                       eps: int,
                       y_array: NDArray[np.float64]) -> NDArray[np.float64]:
        
            starts = np.maximum(key_points - eps, 0)
            ends = np.minimum(key_points + eps, len(y_array))
        
            new_points = np.array([
                start + np.argmin(y_array[start:end])
                for start, end in zip(starts, ends)
            ])
        
            return np.unique(new_points)

        y = self.df["Intensity"]
        x = self.df["Wavenumber"]

        indexes = [i for i in range(len(y))]
        y_array = np.array(y)
        indexes_array = np.array(indexes)

        y_dim: int = np.shape(y_array)[0]
        eps: int = int(0.02 * y_dim)
        best_score_result: int = y_dim
        best_key_points: np.array([], dtype=np.int64)

        for num_points in range(3, 10, 1):
            direct_key_point = direct_method(indexes_array, y_array, num_points)

            fixed_points_without_zero = fix_points(direct_key_point, eps, y_array)
            baseline_without_zero = baseline_on_key_point(
                fixed_points_without_zero,
                y_array[fixed_points_without_zero],
                indexes_array
            )
            score_without_zero = np.sum(y_array < baseline_without_zero)

            points_with_zero = np.unique(np.append(direct_key_point, 0))
            fixed_points_with_zero = fix_points(points_with_zero, eps, y_array)
            baseline_with_zero = baseline_on_key_point(
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

        baseline = baseline_on_key_point(best_key_points, y_for_best_x, indexes_array)

        compensated = self.df["Intensity"] - baseline

        self.df["Compensated"] = compensated - np.min(compensated)
        self.table.setModel(PandasModel(self.df))
        self.plot_canvas.plot(self.df["Wavenumber"], self.df["Compensated"])
        self.current_spectrum_col = "Compensated"


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

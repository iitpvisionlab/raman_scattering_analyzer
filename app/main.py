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

        # Placeholder for compensation logic
        def baseline_on_key_point(x_minima, y_minima, x):
            pchip = PchipInterpolator(x_minima, y_minima)
            baseline = pchip(x)
            return baseline

        y = self.df["Intensity"]
        x = self.df["Wavenumber"]

        num_points = 10  # количество опорных точек

        y_array = np.array(y)
        x_array = np.array(x)

        y_chunks = np.array_split(y_array, num_points)
        x_chunks = np.array_split(x_array, num_points)

        min_y_arr = []
        min_x_arr = []

        for i, (y_chunk, x_chunk) in enumerate(zip(y_chunks, x_chunks)):
            ind_min = np.argmin(y_chunk)
            min_y_arr.append(y_chunk[ind_min])
            min_x_arr.append(x_chunk[ind_min])

        # breakpoint()
        # import matplotlib.pyplot as plt
        baseline = baseline_on_key_point(min_x_arr, min_y_arr, x)
        compensated = self.df["Intensity"] - baseline
        # plt.plot(y, c='darkblue')
        # plt.plot(baseline, c='green')
        # plt.show()

        self.df["Compensated"] = compensated - np.min(compensated)
        self.table.setModel(PandasModel(self.df))
        self.plot_canvas.plot(self.df["Wavenumber"], self.df["Compensated"])
        self.current_spectrum_col = "Compensated"


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

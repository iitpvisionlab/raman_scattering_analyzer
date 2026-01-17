import sys
from typing import TYPE_CHECKING
import os

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
    QMessageBox,
)
from PyQt6.QtCore import (
    QAbstractItemModel,
    QAbstractTableModel,
    Qt,
    QSize,
    QModelIndex,
    pyqtSignal,
    QSettings,
    QDir
)
from PyQt6.QtGui import QAction, QIcon

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

DisplayRole = Qt.ItemDataRole.DisplayRole


class PandasModel(QAbstractTableModel):
    def __init__(self, dataframe: pd.DataFrame):
        super().__init__()
        self._dataframe = dataframe

    def rowCount(self, parent=None):
        return len(self._dataframe)

    def columnCount(self, parent=None):
        return len(self._dataframe.columns)

    def data(self, index, role=DisplayRole):
        if index.isValid() and role == DisplayRole:
            value = self._dataframe.iloc[index.row(), index.column()]
            return str(value)
        return None

    def headerData(self, section, orientation, role=DisplayRole):
        if role == DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return self._dataframe.columns[section]
            else:
                return str(self._dataframe.index[section])
        return None

    def compensate_background(self):
        from .preprocess import compensate_background

        self._dataframe = compensate_background(self._dataframe)
        n_cols = self.columnCount()
        n_rows = self.rowCount()
        self.dataChanged.emit(self.index(0, 1), self.index(n_rows - 1, n_cols - 1))

    def normalize_total(self):
        from .preprocess import normalize_total

        self._dataframe = normalize_total(self._dataframe)
        n_cols = self.columnCount()
        n_rows = self.rowCount()
        self.dataChanged.emit(self.index(0, 1), self.index(n_rows - 1, n_cols - 1))

    def normalize_in_range(self, wavelenght_range: tuple[float, float]):
        from .preprocess import normalize_in_range

        self._dataframe = normalize_in_range(self._dataframe, wavelenght_range)
        n_cols = self.columnCount()
        n_rows = self.rowCount()
        self.dataChanged.emit(self.index(0, 1), self.index(n_rows - 1, n_cols - 1))

    def data_for_columns(self, columns: list[str]) -> pd.DataFrame:
        if not columns:
            return self._dataframe
        return self._dataframe[["Wavenumber"] + columns]
    
    def pca(X, method='centered', n_components=None):
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        if method == 'centered':
            X_processed = X - np.mean(X, axis=0)
            pca = PCA(n_components=n_components)
            scores = pca.fit_transform(X_processed)

        elif method == 'standardized':
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X)
            pca = PCA(n_components=n_components)
            scores = pca.fit_transform(X_processed)

        return {
            'eigenvalues': pca.explained_variance_, # собственные значения
            'percentage_of_variance': pca.explained_variance_ratio_ * 100, # percentage of variance для компонент
            'loadings': pca.components_.T, # собственные векторы
            'scores': scores # координаты объектов в пространстве главных компонент
        }


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

    def plot_all_combined_spectra(self, combined_df: pd.DataFrame):
        self.axes.clear()
        w = combined_df["Wavenumber"]
        for col in combined_df.columns:
            if col != "Wavenumber":
                self.axes.plot(w, combined_df[col], label=col)
        self.axes.set_xlabel("Wavenumber")
        self.axes.set_ylabel("Intensity")
        self.axes.legend()
        self.draw()


class SetRangeDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Set Wavelength Range")
        self.setFixedSize(350, 180)
        layout = QVBoxLayout(self)

        start_layout = QHBoxLayout()
        self.start_edit = QLineEdit(self)
        self.start_edit.setPlaceholderText("Set range start")
        start_layout.addWidget(self.start_edit)
        layout.addLayout(start_layout)

        stop_layout = QHBoxLayout()
        self.stop_edit = QLineEdit(self)
        self.stop_edit.setPlaceholderText("Set range stop")
        stop_layout.addWidget(self.stop_edit)
        layout.addLayout(stop_layout)

        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK", self)
        ok_btn.clicked.connect(self.accept)
        btn_layout.addWidget(ok_btn)
        layout.addLayout(btn_layout)

    def get_range(self):
        return (self.start_edit.text(),
            self.stop_edit.text())
    

class SetNComponents(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Set number of components")
        self.setFixedSize(350, 180)
        layout = QVBoxLayout(self)

        components_layout = QHBoxLayout()
        self.components_edit = QLineEdit(self)
        self.components_edit.setPlaceholderText("Set number of components")
        components_layout.addWidget(self.components_edit)
        layout.addLayout(components_layout)

    def get_n_components(self):
        return self.components_edit.text()


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


class PandasView(QTableView):
    on_plot_requested = pyqtSignal(pd.DataFrame)

    def __init__(self) -> None:
        super().__init__()
        self.spectra = {}

    def set_dataframe(self, dataframe: pd.DataFrame):
        model = PandasModel(dataframe)
        self.setModel(model)
        model.dataChanged.connect(self._replot)
        self.horizontalHeader().selectionModel().selectionChanged.connect(self._replot)
        self._replot()

    def compensate_background(self):
        self.model().compensate_background()

    def normalize_total(self):
        self.model().normalize_total()

    def normalize_in_range(self):
        dialog = SetRangeDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.model().normalize_in_range(dialog.get_range())

    def _replot(self):
        columns = {i.column() for i in self.selectedIndexes()}
        columns.discard(0)
        model = self.model()
        labels = [
            model.headerData(idx, Qt.Orientation.Horizontal, DisplayRole)
            for idx in columns
        ]
        self.on_plot_requested.emit(self.model().data_for_columns(labels))
    
    def load_spectra(self, files: list[str], reset: bool = False):
        if reset:
           self.spectra.clear()
        for f in files:
            df = pd.read_csv(
                f, sep="\t", header=None, names=["Wavenumber", "Intensity"]
            )
            label = os.path.basename(f)
            self.spectra[label] = df
        self.current_spectrum_col = label
        self.combine_spectra_columns()

    def combine_spectra_columns(self):
        merged_df = None
        for label, df in self.spectra.items():
            df_renamed = df.rename(columns={"Intensity": label})
            if merged_df is None:
                merged_df = df_renamed
            else:
                merged_df = pd.merge(
                    merged_df, df_renamed, on="Wavenumber", how="outer"
                )
        merged_df = merged_df.sort_values("Wavenumber").reset_index(drop=True)

        if merged_df.isnull().any(axis=1).any():
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("Warning: Unmatched Wavenumbers")
            msg.setText(
                "Some loaded spectra have uncorresponding wavenumbers.\n"
                "Rows with missing values will be removed.\n"
                "Please click 'OK' to proceed."
            )
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()

        merged_df = merged_df.dropna().reset_index(drop=True)

        self.combined_df = merged_df
        self.set_dataframe(self.combined_df)

    def centered_pca(self):
        dialog = SetNComponents(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.model().pca(self.combined_df, method='centered', n_components=dialog.get_n_components)

    def standardscaler_pca(self):
        dialog = SetNComponents(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.model().pca(self.combined_df, method='standardized', n_components=dialog.get_n_components)

    if TYPE_CHECKING:

        def model(self) -> PandasModel: ...


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings()
        self.setWindowTitle("Spectrum Loader")
        self.setGeometry(100, 100, 800, 600)
        self.table = PandasView()

        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, toolbar)

        load_action = QAction(QIcon.fromTheme(QIcon.ThemeIcon.DocumentNew), "New project", self)
        load_action.triggered.connect(self.new_project)
        toolbar.addAction(load_action)

        load_action = QAction(QIcon.fromTheme(QIcon.ThemeIcon.EditPaste), "Add spectra", self)
        load_action.triggered.connect(self.add_spectra)
        toolbar.addAction(load_action)

        export_action = QAction(QIcon.fromTheme(QIcon.ThemeIcon.DocumentSave), "Export spectrum", self)
        export_action.triggered.connect(self.export_spectrum)
        toolbar.addAction(export_action)

        compensate_action = QAction(QIcon.fromTheme(QIcon.ThemeIcon.WeatherClear), "Compensate background", self)
        compensate_action.triggered.connect(self.table.compensate_background)
        toolbar.addAction(compensate_action)

        compensate_action = QAction(QIcon.fromTheme(QIcon.ThemeIcon.WeatherClear), "Normalize total", self)
        compensate_action.triggered.connect(self.table.normalize_total)
        toolbar.addAction(compensate_action)

        compensate_action = QAction(QIcon.fromTheme(QIcon.ThemeIcon.WeatherClear), "Normalize in range", self)
        compensate_action.triggered.connect(self.table.normalize_in_range)
        toolbar.addAction(compensate_action)

        compensate_action = QAction(QIcon.fromTheme(QIcon.ThemeIcon.WeatherClear), "Centered PCA", self)
        compensate_action.triggered.connect(self.table.centered_pca)
        toolbar.addAction(compensate_action)

        compensate_action = QAction(QIcon.fromTheme(QIcon.ThemeIcon.WeatherClear), "StandardScaler PCA", self)
        compensate_action.triggered.connect(self.table.standardscaler_pca)
        toolbar.addAction(compensate_action)

        # Layout for table and plot (side by side)
        table_plot_layout = QHBoxLayout()
        table_plot_layout.addWidget(self.table)
        self.plot_canvas = PlotCanvas(self)
        table_plot_layout.addWidget(self.plot_canvas)

        # Main layout with button on top
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.addLayout(table_plot_layout)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.current_spectrum_col = None
        self.table.on_plot_requested.connect(self.plot_canvas.plot_all_combined_spectra)

        # self.table.doubleClicked.connect(self.plot_column)

    def add_spectra(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Open Spectrum Files", "", "Text Files (*.dpt);;All Files (*)"
        )
        if not files:
            return
        directory = QDir(files[0]).absolutePath()
        self.settings.setValue("LastDir", directory)
        self.table.load_spectra(files)

    def new_project(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Open Spectrum Files", "", "Text Files (*.dpt);;All Files (*)"
        )
        if not files:
            return
        directory = QDir(files[0]).absolutePath()
        self.settings.setValue("LastDir", directory)
        self.table.load_spectra(files, reset=True)

    def export_spectrum(self):
        labels = [*self.table.spectra.keys()]
        dialog = ExportDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            opts = dialog.get_options()
            export_df = self.table.model().data_for_columns(labels)
            if not opts["filename"].endswith(".dpt"):
                opts["filename"] = opts["filename"] + ".dpt"
            export_df.to_csv(opts["filename"], sep="\t", index=False, header=False)
            if opts["save_image"]:
                img_ext = opts["image_format"].lower()
                img_file = opts["filename"].rsplit(".", 1)[0] + "." + img_ext
                self.plot_canvas.figure.savefig(img_file, format=img_ext)




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

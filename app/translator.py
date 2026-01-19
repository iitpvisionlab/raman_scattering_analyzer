from PyQt6.QtCore import QObject, pyqtSignal
from typing import Dict, Any


class Translator(QObject):

    language_changed = pyqtSignal()
    
    def __init__(self, initial_lang='en'):
        super().__init__()
        self._current_lang = initial_lang
        self._translations: Dict[str, Dict[str, str]] = {
            'en': self._load_english(),
            'ru': self._load_russian()
        }

    def _load_english(self) -> Dict[str, str]:
        return {
            "New project": "New project",
            'Add spectra': 'Add spectra',
            'Export spectrum': 'Export spectrum',
            'Compensate background': 'Compensate background',
            'Normalize total': 'Normalize total',
            'Normalize in range': 'Normalize in range',
            'Centered PCA': 'Centered PCA',
            'StandardScaler PCA': 'StandardScaler PCA',
            'Set range start': 'Set range start',
            'Set range end': 'Set range end',
            'Set number of components': 'Set number of components',
            'Choose filename for spectrum': 'Choose filename for spectrum',
            'Browse...': 'Browse...',
            'Save plot image': 'Save plot image',
            'Spectrum file:': 'Spectrum file:',
            'Image format:': 'Image format:',
            'Export': 'Export',
            'Cancel': 'Cancel',
            "Some loaded spectra have uncorresponding wavenumbers.\n"
            "Rows with missing values will be removed.\n"
            "Please click 'OK' to proceed.": "Some loaded spectra have uncorresponding wavenumbers.\n"
            "Rows with missing values will be removed.\n"
            "Please click 'OK' to proceed.",
            'En': 'En',
        }
    
    def _load_russian(self) -> Dict[str, str]:
        return {
            "New project": 'Новый проект',
            'Add spectra': 'Добавить спектры',
            'Export spectrum': 'Экспортировать спектр',
            'Compensate background': 'Компенсировать фон',
            'Normalize total': 'Нормализовать на полный',
            'Normalize in range': 'Нормализовать в диапазоне',
            'Centered PCA': 'PCA с ковариационной матрицей',
            'StandardScaler PCA': 'PCA с корреляционной матрицей',
            'Set range start': 'Начало диапазона',
            'Set range end': 'Конец диапазона',
            'Set number of components': 'Количество компонент',
            'Choose filename for spectrum': 'Название файла',
            'Browse...': 'Найти...',
            'Save plot image': 'Сохранить изображение графика',
            'Spectrum file:': 'Файл спектра',
            'Image format:': 'Формат изображения:',
            'Export': 'Экспорт',
            'Cancel': 'Отмена',
            "Some loaded spectra have uncorresponding wavenumbers.\n"
            "Rows with missing values will be removed.\n"
            "Please click 'OK' to proceed.": "Загружаемые спектры имеют несоответственные длины волн.\n"
            "Строки с пропущенными значениями будут удалены.\n"
            "Пожалуйста, нажмите 'OK', чтобы продолжить .",
            'En': 'Ру',
        }
    
    def _load_eng_from_rus(self) -> Dict[str, str]:
        return {
            'Новый проект': "New project",
            'Добавить спектры': 'Add spectra',
            'Экспортировать спектр': 'Export spectrum',
            'Компенсировать фон': 'Compensate background',
            'Нормализовать на полный': 'Normalize total',
            'Нормализовать в диапазоне': 'Normalize in range',
            'PCA с ковариационной матрицей': 'Centered PCA',
            'PCA с корреляционной матрицей': 'StandardScaler PCA',
            'Начало диапазона': 'Set range start',
            'Конец диапазона': 'Set range end',
            'Количество компонент': 'Set number of components',
            'Название файла': 'Choose filename for spectrum',
            'Найти...': 'Browse...',
            'Сохранить изображение графика': 'Save plot image',
            'Файл спектра': 'Spectrum file:',
            'Формат изображения:': 'Image format:',
            'Экспорт': 'Export',
            'Отмена': 'Cancel',
            "Загружаемые спектры имеют несоответственные длины волн.\n"
            "Строки с пропущенными значениями будут удалены.\n"
            "Пожалуйста, нажмите 'OK', чтобы продолжить .": "Some loaded spectra have uncorresponding wavenumbers.\n"
            "Rows with missing values will be removed.\n"
            "Please click 'OK' to proceed.",
            'Ру': 'En',
        }
    
    def tr_from_ru(self, key: str) -> str:
        """Translate a key to current language"""
        return self._load_eng_from_rus().get(key, key)
       
    def set_language(self, lang: str):
        """Change language for entire application"""
        if lang in self._translations:
            self._current_lang = lang
            # Emit signal to notify all windows
            self.language_changed.emit()
    
    def get_language(self) -> str:
        return self._current_lang
    
    def tr(self, key: str) -> str:
        """Translate a key to current language"""
        return self._translations.get(self._current_lang, {}).get(key, key)
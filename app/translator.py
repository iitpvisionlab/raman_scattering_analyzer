class Translator:
    _translations = {
        'en': {
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
        },
        'ru': {
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
        }
    }
    
    def __init__(self, lang='en'):
        self.lang = lang
    
    def tr(self, text):
        return self._translations[self.lang].get(text, text)
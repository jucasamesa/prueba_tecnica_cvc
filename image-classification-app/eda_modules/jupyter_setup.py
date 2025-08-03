"""
Script de configuración para Jupyter notebooks.

Este script configura el entorno para usar los módulos EDA en Jupyter notebooks.
Copia y pega este código al inicio de tu notebook.
"""

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib para notebooks
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Añadir módulos EDA al path
# Ajusta la ruta según tu estructura de directorios
current_dir = Path.cwd()
eda_modules_path = current_dir / "scripts" / "eda_modules"

# Si no existe, intentar rutas alternativas
if not eda_modules_path.exists():
    # Intentar desde el directorio padre
    eda_modules_path = current_dir.parent / "scripts" / "eda_modules"
    
if not eda_modules_path.exists():
    # Intentar desde el directorio actual
    eda_modules_path = current_dir / "eda_modules"

if eda_modules_path.exists():
    sys.path.append(str(eda_modules_path))
    print(f"✅ Módulos EDA añadidos al path: {eda_modules_path}")
else:
    print("⚠️  No se encontró el directorio de módulos EDA")
    print("   Asegúrate de que la estructura sea correcta:")
    print("   - scripts/eda_modules/")
    print("   - O eda_modules/ en el directorio actual")

# Importar clases EDA
try:
    from eda_full import EDAFull
    from eda_atomic import EDAAtomic
    from image_analyzer import ImageAnalyzer
    from segmentation import ImageSegmentation
    print("✅ Clases EDA importadas correctamente")
except ImportError as e:
    print(f"❌ Error importando clases EDA: {e}")
    print("   Verifica que las dependencias estén instaladas:")
    print("   pip install opencv-python matplotlib seaborn scikit-image scipy")

# Configurar directorio de datos
# Ajusta según tu estructura
data_dir = current_dir / "data"
if not data_dir.exists():
    data_dir = current_dir.parent / "data"

if data_dir.exists():
    print(f"✅ Directorio de datos encontrado: {data_dir}")
else:
    print(f"⚠️  Directorio de datos no encontrado: {data_dir}")
    print("   Ajusta la variable 'data_dir' según tu estructura")

# Función helper para configurar el entorno
def setup_eda_environment(data_path=None):
    """
    Configura el entorno para análisis EDA.
    
    Args:
        data_path: Ruta al directorio de datos (opcional)
    """
    global data_dir
    
    if data_path:
        data_dir = Path(data_path)
    
    if not data_dir.exists():
        print(f"❌ El directorio de datos no existe: {data_dir}")
        return False
    
    print(f"✅ Entorno configurado:")
    print(f"   • Directorio de datos: {data_dir}")
    print(f"   • Módulos EDA: {eda_modules_path}")
    
    return True

# Función helper para crear instancias de analizadores
def create_analyzers(data_path=None):
    """
    Crea instancias de los analizadores EDA.
    
    Args:
        data_path: Ruta al directorio de datos (opcional)
    
    Returns:
        tuple: (eda_full, eda_atomic) o (None, None) si hay error
    """
    if not setup_eda_environment(data_path):
        return None, None
    
    try:
        eda_full = EDAFull(data_dir)
        eda_atomic = EDAAtomic(data_dir)
        print("✅ Analizadores EDA creados correctamente")
        return eda_full, eda_atomic
    except Exception as e:
        print(f"❌ Error creando analizadores: {e}")
        return None, None

# Función helper para verificar dependencias
def check_dependencies():
    """
    Verifica que todas las dependencias estén instaladas.
    """
    dependencies = [
        'cv2',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'skimage',
        'scipy'
    ]
    
    missing = []
    for dep in dependencies:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)
    
    if missing:
        print(f"❌ Dependencias faltantes: {missing}")
        print("   Instala con: pip install " + " ".join(missing))
        return False
    else:
        print("✅ Todas las dependencias están instaladas")
        return True

# Función helper para mostrar información del dataset
def show_dataset_info(eda_full):
    """
    Muestra información básica del dataset.
    
    Args:
        eda_full: Instancia de EDAFull
    """
    try:
        eda_full.load_datasets()
        
        print("📊 Información del Dataset:")
        print(f"   • Registros totales: {len(eda_full.dataset_df)}")
        print(f"   • Columnas: {list(eda_full.dataset_df.columns)}")
        
        if eda_full.downloaded_df is not None:
            print(f"   • Imágenes descargadas: {len(eda_full.downloaded_df)}")
            successful = eda_full.downloaded_df['download_success'].sum()
            print(f"   • Descargas exitosas: {successful}")
        
        # Información del target
        target_counts = eda_full.dataset_df['correct_background?'].value_counts()
        print(f"   • Distribución del target:")
        for value, count in target_counts.items():
            percentage = (count / len(eda_full.dataset_df)) * 100
            print(f"     - {value}: {count} ({percentage:.1f}%)")
        
        # Información por sitio
        site_counts = eda_full.dataset_df['site_id'].value_counts()
        print(f"   • Distribución por sitio:")
        for site, count in site_counts.head(5).items():
            print(f"     - {site}: {count}")
        
        return True
    except Exception as e:
        print(f"❌ Error mostrando información del dataset: {e}")
        return False

# Ejecutar verificación de dependencias
print("\n🔍 Verificando dependencias...")
check_dependencies()

print("\n📋 Configuración completada!")
print("   Para usar los módulos EDA:")
print("   1. Ajusta 'data_dir' si es necesario")
print("   2. Usa create_analyzers() para crear instancias")
print("   3. Usa show_dataset_info() para ver información del dataset")

# Ejemplo de uso:
print("\n💡 Ejemplo de uso:")
print("""
# Crear analizadores
eda_full, eda_atomic = create_analyzers()

# Mostrar información del dataset
show_dataset_info(eda_full)

# Análisis completo
quality_df = eda_full.analyze_image_quality(sample_size=50)

# Análisis atómico
eda_atomic.visualize_random_sample(n_samples=3)
""") 

from pathlib import Path
from importlib.metadata import version
from importlib import import_module
import platform
import sys


# ==========================================
# 1. CONFIGURATION DU PROJET
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

# Liste des bibliotheques a verifier.
# Nom de la distribution : nom du module Python.

LIBRARIES = {

    "numpy": "numpy",

    "pandas": "pandas",

    "scikit-learn": "sklearn",

    "matplotlib": "matplotlib",

    "seaborn": "seaborn",

    "pymupdf": "pymupdf",

    "pytest": "pytest"
}


# ==========================================
# 2. VERIFICATION DE PYTHON
# ==========================================

def check_python():

    print("\n===== PYTHON ENVIRONMENT =====")

    print(f"Python version: {platform.python_version()}")

    print(f"Operating system: {platform.system()}")

    print(f"Architecture: {platform.machine()}")

    print(f"Python executable: {sys.executable}")

    # Verifier si Python utilise
    # un environnement virtuel.

    is_virtual_env = (
        sys.prefix != sys.base_prefix
    )

    print(f"Virtual environment: {is_virtual_env}")

    return is_virtual_env


# ==========================================
# 3. VERIFICATION DES LIBRAIRIES
# ==========================================

def check_libraries():

    print("\n===== LIBRARY VERIFICATION =====")

    success = True

    for package, module in LIBRARIES.items():

        try:

            # Verifier que le module peut
            # effectivement etre importe.

            import_module(module)

            # Recuperer sa version installee.

            package_version = version(package)

            print(
                f"[OK] {package:<20} "
                f"{package_version}"
            )

        except Exception as error:

            print(f"[ERROR] {package}: {error}")

            success = False

    return success


# ==========================================
# 4. VERIFICATION DES DOSSIERS
# ==========================================

def check_directories():

    print("\n===== PROJECT STRUCTURE =====")

    directories = [

        "data/raw",

        "data/processed",

        "data/samples",

        "ai_engine/extraction",

        "ai_engine/preprocessing",

        "ai_engine/nlp",

        "ai_engine/matching",

        "ai_engine/evaluation",

        "ai_engine/models",

        "ai_engine/api",

        "backend",

        "frontend",

        "reports",

        "tests",

        "docs"
    ]

    success = True

    for directory in directories:

        path = BASE_DIR / directory

        if path.is_dir():

            print(f"[OK] {directory}")

        else:

            print(f"[MISSING] {directory}")

            success = False

    return success


# ==========================================
# 5. FONCTION PRINCIPALE
# ==========================================

def main():

    print("=" * 60)

    print("SMARTCV AI - Environment Verification")

    print("=" * 60)

    # Verifier les trois composants.

    python_ok = check_python()

    libraries_ok = check_libraries()

    directories_ok = check_directories()

    # Afficher le resultat global.

    print("\n" + "=" * 60)

    print("ENVIRONMENT SUMMARY")

    print("=" * 60)

    print(f"Virtual environment: {python_ok}")

    print(f"Libraries installed: {libraries_ok}")

    print(f"Project structure: {directories_ok}")

    success = (

        python_ok

        and libraries_ok

        and directories_ok
    )

    if success:

        print("\nSmartCV AI environment is ready!")

        print("All checks passed successfully.")

    else:

        print("\nEnvironment verification failed.")

        print("Please check the errors above.")

    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":

    sys.exit(main())

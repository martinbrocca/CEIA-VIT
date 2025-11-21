# Motor de Búsqueda Multimodal de Recetas

Sistema de búsqueda de recetas que combina procesamiento de lenguaje natural y visión por computadora para encontrar recetas por texto, ingredientes o imágenes, con filtros dietéticos inteligentes y sustituciones de ingredientes.

##  Características

-  **Búsqueda Semántica**: Búsqueda sobre 231K recetas usando embeddings de texto
-  **Búsqueda por Imagen**: Sube fotos de platos para encontrar recetas similares usando CLIP
-  **Filtros Dietéticos**: Vegetariano, vegano, sin gluten, sin lácteos
-  **Sustituciones Inteligentes**: Sugerencias de ingredientes alternativos según preferencias dietéticas
-  **Retrieval Rápido**: Búsqueda en milisegundos usando índices FAISS
-  **Tracking MLOps**: Experimentos registrados en MLflow/Databricks

##  Arquitectura del Proyecto
```
CEIA-VIT/
├── app/                              # Interfaz Streamlit
│   ├── streamlit_app.py             # App principal
│   ├── strings_es.py                # Traducciones al español
│   ├── components/                  # Componentes UI reutilizables
│   └── assets/                      # Imágenes, CSS
│
├── src/                             # Código fuente principal
│   ├── data/                        # Preprocesamiento de datos
│   │   ├── preprocessing.py         # Limpieza y preparación de recetas
│   │   └── dietary_tagger.py        # Clasificación dietética
│   │
│   ├── models/                      # Modelos de embeddings y retrieval
│   │   ├── embeddings.py            # Sentence-transformers embeddings
│   │   ├── clip_embeddings.py       # CLIP embeddings multimodales
│   │   ├── retrieval.py             # Sistema de búsqueda por texto
│   │   └── clip_retrieval.py        # Sistema de búsqueda por imagen
│   │
│   ├── substitutions/               # Motor de sustituciones
│   │   ├── rules.py                 # Lógica de sustituciones
│   │   └── substitution_db.json     # Base de datos de sustituciones
│   │
│   ├── evaluation/                  # Scripts de evaluación
│   │   ├── hard_negatives.py        # Tests de negativos difíciles
│   │   └── clip_image_tests.py      # Tests de pares confusos
│   │
│   └── utils/                       # Utilidades
│       ├── config.py                # Configuración de rutas
│       ├── device.py                # Manejo de GPU/CPU/MPS
│       └── mlflow_logger.py         # Logging a MLflow
│
├── data/                            # Datos (gitignored)
│   ├── raw/food-com/               # Dataset original Food.com
│   ├── processed/                   # Recetas preprocesadas
│   └── embeddings/                  # Embeddings y índices FAISS
│
├── notebooks/                       # Jupyter notebooks para exploración
├── configs/                         # Archivos de configuración
├── mlruns/                         # Experimentos MLflow locales
└── tests/                          # Tests unitarios
```

##  Instalación

### Requisitos
- Python 3.10+
- GPU recomendada (funciona en CPU/MPS también)
- ~4GB de espacio en disco para datos y embeddings

### Pasos

1. **Clonar el repositorio**
```bash
git clone <repo-url>
cd CEIA-VIT
```

2. **Crear entorno virtual e instalar dependencias**
```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

3. **Descargar el dataset Food.com**
```bash
# Requiere Kaggle API configurada
cd data/raw/food-com
kaggle datasets download -d shuyangli94/food-com-recipes-and-user-interactions
unzip food-com-recipes-and-user-interactions.zip
rm food-com-recipes-and-user-interactions.zip
cd ../../..
```

4. **Configurar variables de entorno (opcional para Databricks)**
```bash
# Crear .env en la raíz del proyecto
cat > .env << EOF
DATABRICKS_HOST=https://tu-workspace.cloud.databricks.com
DATABRICKS_TOKEN=tu_token_de_acceso
DATABRICKS_USER=tu.email@dominio.com
EOF
```

##  Pipeline de Datos

### 1. Preprocesamiento
```bash
python src/data/preprocessing.py
```
- Limpia y estructura 231K recetas
- Genera texto enriquecido para embeddings
- Registra métricas en MLflow

### 2. Crear Embeddings (Sentence-Transformers)
```bash
python src/models/embeddings.py
```
- Modelo: `sentence-transformers/all-MiniLM-L6-v2`
- Genera embeddings de 384 dimensiones
- ~20 segundos en GPU RTX 5090

### 3. Crear Embeddings CLIP (Multimodal)
```bash
python src/models/clip_embeddings.py
```
- Modelo: `openai/clip-vit-base-patch32`
- Genera embeddings de 512 dimensiones
- Permite búsqueda por imagen

##  Uso

### Lanzar la Aplicación Web
```bash
streamlit run app/streamlit_app.py
```

La app se abrirá en `http://localhost:8501` con tres modos de búsqueda:

1. ** Búsqueda por Imagen**: Sube una foto de un plato
2. ** Búsqueda por Texto**: Describe lo que buscas
3. ** Búsqueda por Ingredientes**: Lista ingredientes disponibles

### Evaluación del Sistema

**Tests de Negativos Difíciles:**
```bash
# Evaluar Sentence-Transformers
python src/evaluation/hard_negatives.py

# Evaluar CLIP
python src/evaluation/hard_negatives.py --clip
```

**Tests de Pares Confusos (tipo acelga/espinaca):**
```bash
python src/evaluation/clip_image_tests.py
```

##  Resultados de Evaluación

### Comparación de Modelos

| Métrica | Sentence-Transformers | CLIP |
|---------|----------------------|------|
| **Pass Rate** | 60% (3/5) | 60% (3/5) |
| **Precisión Objetivo** | 80% | 72% |
| **Similitud Promedio** | 0.687 | 0.826 |
| **Tiempo Búsqueda** | 36.2ms | 38.9ms |

**Conclusión**: CLIP muestra mayor confianza en las coincidencias (similitud +20%), ideal para búsqueda por imagen, con overhead mínimo de latencia.

##  Stack Tecnológico

- **Embeddings**: Sentence-Transformers, CLIP (HuggingFace)
- **Búsqueda**: FAISS (Facebook AI Similarity Search)
- **UI**: Streamlit
- **MLOps**: MLflow + Databricks
- **Procesamiento**: pandas, NumPy
- **DL**: PyTorch (con soporte CUDA/MPS/CPU)

##  MLflow/Databricks

El pipeline registra automáticamente:
- Estadísticas de preprocesamiento
- Métricas de embeddings (tiempo, dimensión, throughput)
- Resultados de evaluación (precision, recall, contamination rate)

Ver experimentos localmente:
```bash
mlflow ui
# Abre http://localhost:5000
```

O en Databricks (si está configurado):
- Los experimentos aparecen en **Machine Learning → Experiments**

##  Trabajo Futuro

### Mejoras Planificadas
- **Sustituciones con LLM**: Usar Ollama/Groq para sustituciones contextuales en lugar de reglas estáticas
- **Dataset Recipe1M+**: Integrar cuando esté disponible para búsqueda imagen→receta real
- **Fine-tuning**: Ajustar CLIP en pares imagen-receta específicos
- **Feedback de Usuario**: Sistema de rating para mejorar el ranking

### Extensiones Posibles
- API REST con FastAPI
- Caché de embeddings con Redis
- Sistema de recomendación personalizado
- Soporte multiidioma (traducción de recetas)

##  Dataset

**Food.com (Kaggle)**
- 231,637 recetas
- Ingredientes, pasos, tags, metadatos nutricionales
- [Descargar aquí](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)

## 🎓 Contexto Académico

Proyecto desarrollado para **CEIA (Especialización en Inteligencia Artificial)** - Universidad de Buenos Aires

**Materia**: Visión por Computadora III

**Objetivo**: Demostrar capacidades de búsqueda multimodal combinando NLP y CV con MLOps



## 📄 Licencia

Este proyecto es con fines educativos.

---

**💡 Tip**: Para profesores evaluando este proyecto - todos los comandos funcionan sin configuración adicional excepto las credenciales de Databricks (opcional). El sistema usa embeddings pre-calculados y puede ejecutarse completamente en modo local.
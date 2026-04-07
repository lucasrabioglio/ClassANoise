#!/usr/bin/env bash
# reorganizar_repo.sh
# Reorganiza el repositorio en la estructura propuesta.
# Ejecutar desde la raíz del repositorio.

set -euo pipefail

echo "Creando directorios..."
mkdir -p core
mkdir -p estimators
mkdir -p simulation
mkdir -p performance/tests
mkdir -p notebooks/analisis
mkdir -p notebooks/simulaciones
mkdir -p notebooks/comparaciones

echo "Moviendo archivos de core/..."
mv -v ClassA.py            core/
mv -v func_min.py          core/
mv -v momento.py           core/
mv -v Nelder_Mead.py       core/

echo "Moviendo estimadores..."
mv -v Estimadores.py                    estimators/
mv -v Estimadores_terminos_suma.py      estimators/
mv -v Estimador_CDF.py                  estimators/
mv -v est_inicial.py                    estimators/
mv -v est_inicial_densidad.py           estimators/
mv -v pdf_teorica.py                    estimators/
mv -v pdf_CDF_teorica_experimental.py   estimators/
mv -v RFI_EMTwoParamEst.py             estimators/
mv -v RFI_EMParamA.py                  estimators/
mv -v RFI_EMParamK_AFixed.py           estimators/
mv -v RFI_EMCalculateObjFunc.py        estimators/

echo "Moviendo simulaciones..."
mv -v simulacion_est_complejo_calidad.py        simulation/
mv -v simulacion_est_complejo_calidad_zoomed.py simulation/
mv -v simulacion_est_inicial_calidad.py         simulation/
mv -v simulacion_est_Luc_calidad.py             simulation/
mv -v muestras_potencia_log.py                  simulation/
mv -v histograma_potencia.py                    simulation/
mv -v variacion_terminos_sumatoria.py           simulation/

echo "Moviendo performance/..."
mv -v Performance/Performance_Estimadores_MSE.py          performance/
mv -v Performance/Performance_Estimadores_Tiempos.py      performance/
mv -v Performance/Performance_Estimadores_Tiempos_2.py    performance/
mv -v Performance/Performance_Estimador_Propio.py         performance/
mv -v Performance/Performance_Estimador_inicial_ESL.py    performance/
mv -v Performance/performance_est_inicial.py              performance/
mv -v Performance/performance_est_inicial2.py             performance/

echo "Moviendo tests (func_min_*)..."
mv -v Tests/func_min_*.py performance/tests/

echo "Moviendo notebooks de análisis..."
mv -v Indices_AKAIKE.ipynb      notebooks/analisis/
mv -v indices_AKAIKE_auto.ipynb notebooks/analisis/

echo "Moviendo notebooks de simulaciones..."
mv -v distribucion_estimaciones.ipynb                      notebooks/simulaciones/
mv -v sim_densidad_alfa_ESL.ipynb                          notebooks/simulaciones/
mv -v simulacion_est_simple_variacion_gamma_ESL.ipynb      notebooks/simulaciones/
mv -v simulacion_est_complejo_calidad.ipynb                notebooks/simulaciones/ 2>/dev/null || true
mv -v performance_estimadores_zona_gaussiana_ESL.ipynb     notebooks/simulaciones/ 2>/dev/null || true

echo "Moviendo notebooks de comparaciones..."
mv -v comparacion_estSimple_vs_INEMM.ipynb notebooks/comparaciones/

echo "Moviendo notebooks de performance (si existen en Performance/)..."
mv -v Performance/Performance_Estimador_inicial_ESL.ipynb  notebooks/analisis/ 2>/dev/null || true

echo "Eliminando directorios originales si quedaron vacíos..."
rmdir Performance 2>/dev/null && echo "  Eliminado: Performance/" || echo "  Performance/ no estaba vacío (revisar manualmente)"
rmdir Tests      2>/dev/null && echo "  Eliminado: Tests/"       || echo "  Tests/ no estaba vacío (revisar manualmente)"

echo ""
echo "Listo. Estructura resultante:"
find . -not -path '*/.git/*' | sort

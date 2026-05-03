@echo off
echo ========================================================
echo   AI Oracle Pick 3 - Actualizador de Datos (Titan)
echo ========================================================
echo.
python scrape.py
echo.
if exist data_scraped.csv (
    echo [EXITO] Los datos se han guardado en data_scraped.csv
    echo Abre el programa e importa el archivo CSV para actualizar.
) else (
    echo [ERROR] No se pudo obtener la informacion. Revisa tu conexion.
)
echo.
pause

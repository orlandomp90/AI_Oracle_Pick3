@echo off
setlocal
title AI ORACLE v4.0 - UNIFIED CONSOLE
color 0B

:: Limpiar pantalla para un inicio fresco
cls

echo ======================================================
echo   AI ORACLE v4.0 - SISTEMA UNIFICADO (SINGLE CONSOLE)
echo ======================================================
echo.
echo [1/2] Iniciando Backend de Python (Port 5001)...
:: 'start /b' corre el comando en la misma ventana, permitiendo ver todos los logs juntos
start /b python api.py

:: Pequeña pausa para asegurar que el server de Python inicie
timeout /t 2 >nul

echo.
echo [2/2] Iniciando Servidor Web Node (Port 5000)...
echo ------------------------------------------------------
echo Servidor Web: http://localhost:5000
echo Engine IA:    http://localhost:5001
echo ------------------------------------------------------
echo.

:: Abrir el navegador automáticamente
start http://localhost:5000

echo [READY] El sistema esta listo. 
echo [INFO] Los logs de Python y Node se mezclaran abajo:
echo [INFO] Presiona Ctrl+C en esta ventana para detener ambos.
echo ======================================================
echo.

:: Iniciar Node en primer plano para mantener la consola abierta
node server.js

:: Al terminar, mostramos mensaje de salida
echo.
echo [SISTEMA] Programa finalizado. 
echo [HINT] Si el backend sigue corriendo en segundo plano, cierra esta ventana.
pause

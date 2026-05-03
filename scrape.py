import requests
from bs4 import BeautifulSoup
import csv
import re

print(">>> Iniciando Scraper de Florida Pick 3...")
url = "https://www.lotteryusa.com/florida/pick-3/"

try:
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    
    draws = []
    # Buscamos secciones de resultados recientes
    # El patrón busca una fecha y luego intenta capturar los 3 dígitos siguientes en la estructura HTML
    sections = soup.find_all(['div', 'tr', 'li', 'ul'])
    
    # Intento 1: Regex robusto sobre el texto plano
    text_full = soup.get_text(separator=' ', strip=True)
    # Buscamos fechas tipo "Apr 1, 2026" seguidas de números
    date_regex = r'([A-Z][a-z]{2}\s\d{1,2},\s\d{4})'
    digit_regex = r'\b(\d)\b'
    
    matches = list(re.finditer(date_regex, text_full))
    for i, match in enumerate(matches):
        date_str = match.group(1)
        # Buscar en el texto posterior a la fecha los primeros 3 dígitos
        after_text = text_full[match.end():]
        digits = re.findall(digit_regex, after_text)
        if len(digits) >= 3:
            draws.append([date_str, digits[0], digits[1], digits[2]])
            if len(draws) > 20: break # Limitar a los más recientes
    
    # Eliminar duplicados manteniendo orden
    seen = set()
    unique_draws = []
    for d in draws:
        t = tuple(d)
        if t not in seen:
            seen.add(t)
            unique_draws.append(d)

    if not unique_draws:
        print(">>> [Aviso] No se pudieron parsear resultados con el nuevo formato HTML.")
        print(">>> Usando modo Fallback (Por favor usa la opción de subir CSV manual en la UI).")
    else:
        with open('data_scraped.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Fecha', 'N1', 'N2', 'N3'])
            writer.writerows(unique_draws)
        print(f">>> ¡Éxito! Se guardaron {len(unique_draws)} sorteos en 'data_scraped.csv'.")

except Exception as e:
    print(f">>> Error crítico durante el scraping: {e}")


const http = require('http');
const fs = require('fs');
const path = require('path');
const https = require('https');
const PORT = 5000;

const MIME_TYPES = {
    '.html':'text/html; charset=utf-8', '.js':'text/javascript; charset=utf-8',
    '.css':'text/css; charset=utf-8', '.json':'application/json',
    '.png':'image/png', '.jpg':'image/jpeg', '.jpeg':'image/jpeg',
    '.gif':'image/gif', '.svg':'image/svg+xml', '.ico':'image/x-icon',
    '.pdf':'application/pdf', '.csv':'text/csv',
    '.woff':'font/woff', '.woff2':'font/woff2', '.ttf':'font/ttf', '.map':'application/json'
};

function fetchPage(url) {
    return new Promise((resolve, reject) => {
        const headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36' };
        https.get(url, { headers }, (res) => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => resolve(data));
        }).on('error', reject);
    });
}

function parseDraws(html, suffix) {
    const draws = [];
    // Regex robusto que busca fechas seguidas de 3 dígitos (Pick 3)
    // Busca patrones como "Oct 12, 2023" o "10/12/2023" y luego captura los 3 números siguientes
    const regex = /(?:>|\b)([A-Z][a-z]{2}\s\d{1,2},\s\d{4}|\d{1,2}\/\d{1,2}\/\d{2,4})[\s\S]*?(?:span|div|li)[^>]*>(\d)<\/(?:span|div|li)>[\s\S]*?(?:span|div|li)[^>]*>(\d)<\/(?:span|div|li)>[\s\S]*?(?:span|div|li)[^>]*>(\d)<\/(?:span|div|li)>/g;
    
    let match;
    while ((match = regex.exec(html)) !== null) {
        const localArea = html.substring(Math.max(0, match.index - 100), match.index + 150).toLowerCase();
        let inferredSuffix = suffix || "";
        if (!suffix) {
            if (localArea.includes('eve') || localArea.includes('night')) inferredSuffix = "EVE";
            else if (localArea.includes('mid') || localArea.includes('day')) inferredSuffix = "MD";
        }
        
        draws.push({
            date: match[1].trim() + (inferredSuffix ? ` (${inferredSuffix})` : ""),
            n1: parseInt(match[2]),
            n2: parseInt(match[3]),
            n3: parseInt(match[4])
        });
    }
    
    // Fallback: Si no encontró nada, buscar formato de lista simple (como en el markdown)
    if (draws.length === 0) {
        const fallbackRegex = /([A-Z][a-z]{2}\s\d{1,2},\s\d{4})[\s\S]*?(\d)\s+(\d)\s+(\d)/g;
        while ((match = fallbackRegex.exec(html)) !== null) {
            const localArea = html.substring(Math.max(0, match.index - 100), match.index + 150).toLowerCase();
            let inferredSuffix = suffix || "";
            if (!suffix) {
                if (localArea.includes('eve') || localArea.includes('night')) inferredSuffix = "EVE";
                else if (localArea.includes('mid') || localArea.includes('day')) inferredSuffix = "MD";
            }
            
            draws.push({
                date: match[1].trim() + (inferredSuffix ? ` (${inferredSuffix})` : ""),
                n1: parseInt(match[2]),
                n2: parseInt(match[3]),
                n3: parseInt(match[4])
            });
        }
    }
    
    return draws;
}

const server = http.createServer(async (req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }
    if (req.url === '/favicon.ico') { res.writeHead(204); res.end(); return; }
    if (req.url.startsWith('/.well-known/')) { res.writeHead(404); res.end(); return; }

    // ═══════════════════════════════════════
    // API: MULTI-SOURCE SCRAPER (Midday + Evening)
    // ═══════════════════════════════════════
    if (req.url === '/api/scrape') {
        console.log('[API] Multi-source scrape starting...');
        try {
            const allDraws = [];

            // Florida Pick 3 Hub (Latest)
            try {
                const hubHtml = await fetchPage('https://www.lotteryusa.com/florida/pick-3/');
                const hubDraws = parseDraws(hubHtml); // El suffix se sacará de los datos si es posible, o se inferirá
                allDraws.push(...hubDraws);
                console.log(`[API] Main Hub: ${hubDraws.length} draws`);
            } catch (e) { console.warn('[API] Hub fetch failed:', e.message); }

            // Archive (Previous Draws)
            try {
                const pastHtml = await fetchPage('https://www.lotteryusa.com/florida/pick-3/year');
                const pastDraws = parseDraws(pastHtml);
                const existingDates = new Set(allDraws.map(d => d.date));
                const newPast = pastDraws.filter(d => !existingDates.has(d.date));
                allDraws.push(...newPast);
                console.log(`[API] Archive: ${newPast.length} new draws`);
            } catch (e) { console.warn('[API] Archive fetch failed:', e.message); }

            console.log(`[API] Total: ${allDraws.length} draws`);
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ success: true, count: allDraws.length, draws: allDraws.slice(0, 200) }));
        } catch (err) {
            console.error('[API] Scrape error:', err.message);
            res.writeHead(500, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ success: false, error: err.message }));
        }
        return;
    }

    // ═══════════════════════════════════════
    // STATIC FILE SERVER
    // ═══════════════════════════════════════
    const cleanUrl = req.url.split('?')[0];
    let filePath = path.join(__dirname, cleanUrl === '/' ? 'index.html' : cleanUrl);

    if (!filePath.startsWith(__dirname)) { res.writeHead(403); res.end('Denied'); return; }

    const ext = path.extname(filePath).toLowerCase();
    const ct = MIME_TYPES[ext] || 'application/octet-stream';

    fs.readFile(filePath, (error, content) => {
        if (error) {
            if (error.code === 'ENOENT') {
                console.warn(`[404] ${filePath}`);
                res.writeHead(404, { 'Content-Type': 'text/plain' });
                res.end(`404 - Not found: ${cleanUrl}`);
            } else {
                res.writeHead(500, { 'Content-Type': 'text/plain' });
                res.end('Server Error: ' + error.code);
            }
        } else {
            console.log(`[200] ${req.method} ${cleanUrl}`);
            res.writeHead(200, { 'Content-Type': ct });
            res.end(content);
        }
    });
});

server.listen(PORT, () => {
    console.log('═══════════════════════════════════════════════════');
    console.log(`  AI ORACLE TITAN v3.0 - PORT ${PORT}`);
    console.log(`  http://localhost:${PORT}`);
    console.log('  Multi-Source: Midday + Evening + Past Results');
    console.log('═══════════════════════════════════════════════════');
});

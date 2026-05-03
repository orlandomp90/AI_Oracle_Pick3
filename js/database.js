/**
 * Modulo de Base de Datos - Wrapper sobre IndexedDB
 */
const DB_NAME = 'FloridaPick3DB';
const DB_VERSION = 1;
const STORE_NAME = 'draws';

class DatabaseManager {
    constructor() {
        this.db = null;
    }

    async init() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(DB_NAME, DB_VERSION);

            request.onerror = (event) => {
                console.error("Error al abrir IndexedDB:", event.target.error);
                reject("Database error");
            };

            request.onsuccess = (event) => {
                this.db = event.target.result;
                resolve(this.db);
            };

            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                if (!db.objectStoreNames.contains(STORE_NAME)) {
                    const store = db.createObjectStore(STORE_NAME, { keyPath: 'id', autoIncrement: true });
                    store.createIndex('date', 'date', { unique: false });
                }
            };
        });
    }

    async addDraws(draws) {
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([STORE_NAME], 'readwrite');
            const store = transaction.objectStore(STORE_NAME);

            draws.forEach(draw => {
                store.add(draw);
            });

            transaction.oncomplete = () => resolve();
            transaction.onerror = (e) => reject(e);
        });
    }

    async deleteFullDatabase() {
        if (this.db) {
            this.db.close();
        }
        return new Promise((resolve, reject) => {
            const request = indexedDB.deleteDatabase(DB_NAME);
            request.onsuccess = () => resolve();
            request.onerror = (e) => reject(e);
            request.onblocked = () => {
                console.warn("Borrado bloqueado: Cierra otras pestañas del programa.");
                resolve(); // Proceder igualmente o informar al usuario
            };
        });
    }

    async getAllDraws() {
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([STORE_NAME], 'readonly');
            const store = transaction.objectStore(STORE_NAME);
            const request = store.getAll();

            request.onsuccess = () => {
                let result = request.result;
                result.sort((a,b) => {
                    const parseDateStr = (dateStr) => {
                        if (!dateStr) return 0;
                        let cleanStr = dateStr.replace(/\(.*?\)/g, '').trim(); // Eliminar (MD) o (EVE)
                        let d = new Date(cleanStr);
                        if (!isNaN(d.getTime())) return d.getTime();
                        
                        // Fallback para dd/mm/yy numérico estricto
                        const p = cleanStr.split(/[\/\-]/);
                        if(p.length >= 3) { 
                            let y = parseInt(p[2]); if(y < 100) y += 2000; 
                            return new Date(y, parseInt(p[1])-1, parseInt(p[0])).getTime(); 
                        }
                        return 0;
                    };
                    
                    const da = parseDateStr(a.date);
                    const db = parseDateStr(b.date);
                    
                    if (da && db && da !== db) return da - db;
                    
                    // Empate => Ordernar por turno
                    const sa = (a.date||'').toLowerCase().includes('eve') ? 1 : 0;
                    const sb = (b.date||'').toLowerCase().includes('eve') ? 1 : 0;
                    return sa - sb;
                });
                resolve(result);
            };
            request.onerror = (e) => reject(e);
        });
    }
}

const dbManager = new DatabaseManager();

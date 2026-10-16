# Latência e vazão da inferência

- Ambiente: Python 3.12.10 on Linux | 12 cores | load average 6.44, 8.65, 8.49
- Carga do modelo a frio: 57.4 ms (cancer_classifier.joblib)
- Agregação: mediana de 3 execução(ões) por cenário
- Orçamento de latência: 100 ms (p95)

### model (direct)

| Concurrency | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (req/s) |
|-------------|----------|----------|----------|--------------------|
| 1 | 0.45 | 0.80 | 0.94 | ~1915 |
| 2 | 0.43 | 1.00 | 15.57 | ~1878 |
| 4 | 0.72 | 7.52 | 50.43 | ~1399 |
| 8 | 0.64 | 6.80 | 50.54 | ~1535 |
| 16 | 0.68 | 1.66 | 141.60 | ~1406 |

### api (/predict)

| Concurrency | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (req/s) |
|-------------|----------|----------|----------|--------------------|
| 1 | 3.50 | 4.54 | 5.03 | ~284 |
| 2 | 14.49 | 25.94 | 28.35 | ~130 |
| 4 | 30.82 | 51.01 | 60.96 | ~127 |
| 8 | 19.52 | 33.33 | 51.93 | ~375 |
| 16 | 33.22 | 56.77 | 75.61 | ~449 |

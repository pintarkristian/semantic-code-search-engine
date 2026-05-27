# Sample curl Commands

```bash
# Liveness and readiness
curl http://localhost:8000/health

# Version and active index manifest
curl http://localhost:8000/version

# Prometheus metrics
curl http://localhost:8000/metrics

# Start indexing a repository
curl -X POST http://localhost:8000/index \
  -H "content-type: application/json" \
  -d '{"repo_path": "/path/to/repo", "rebuild": false}'

# Search the loaded index
curl "http://localhost:8000/search?q=validate+JWT+token&k=5"
curl "http://localhost:8000/search?q=extract+user+id+from+token&k=5"
```

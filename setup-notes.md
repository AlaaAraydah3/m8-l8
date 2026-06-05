# Setup Notes — Module 8 Lab

## Environment

- Python 3.10+
- Weaviate running via Docker (see below)
- `sentence-transformers`, `weaviate-client` installed via `pip install -r requirements.txt`

## Weaviate Docker

```bash
docker run -d \
  --name weaviate \
  -p 8080:8080 \
  -e QUERY_DEFAULTS_LIMIT=25 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
  -e DEFAULT_VECTORIZER_MODULE=none \
  -e CLUSTER_HOSTNAME=node1 \
  semitechnologies/weaviate:1.24.1
```

Verify it's running:
```bash
curl http://localhost:8080/v1/.well-known/ready
# → {"status":"READY"}
```

## Common Issues

- **`BatchNotStartedError`**: Use `with client.batch as batch:` rather than calling `client.batch.flush()` manually; the context manager handles flushing.
- **`numpy array not JSON serializable`**: Always call `.tolist()` on numpy vectors before passing to `batch.add_data_object(vector=...)`.
- **Stale class after schema change**: `create_schema` deletes and recreates the class — safe to re-run, but all previously ingested data is wiped.
- **BM25 returns empty results**: Make sure `indexSearchable: True` is set on `title`, `question_text`, and `answer_text` — NOT on `text` (that field is stored-only).
- **Model download slow on first run**: `all-MiniLM-L6-v2` is ~90 MB; cached in `~/.cache/huggingface` after first download.

## CI / GitHub Actions

- Enable Actions on your fork before pushing: Actions tab → "I understand my workflows, go ahead and enable them".
- If the autograder shows "Workflows are disabled", push an empty commit to re-trigger:
  ```bash
  git commit --allow-empty -m "ci: trigger"
  git push origin lab-vector-retrieval
  ```
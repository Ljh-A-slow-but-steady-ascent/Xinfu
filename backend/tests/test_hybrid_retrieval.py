from __future__ import annotations

import sys
import time

import pytest

from backend.app.services.rag.retriever import BM25Index, LocalRagRetriever, RagModelClient, _canonical_source


def test_bm25_uses_idf_and_length_normalization():
    index = BM25Index([
        ['common', 'rare'],
        ['common'] * 20 + ['rare'],
        ['common', 'other'],
    ])
    rare = index.score('rare')
    assert rare[0] > rare[1] > rare[2] == 0
    assert index.idf['rare'] > index.idf['common']


def test_search_candidates_are_not_evidence_verdicts(monkeypatch):
    monkeypatch.setenv('DERMAGUARD_RAG_ADVANCED', '0')
    retriever = LocalRagRetriever()
    results = retriever.search('北京天气变化怎么预测')
    assert results  # An accidental word overlap remains a candidate, not truth.
    assert all(item['evidence_status'] == '待核验' for item in results)
    assert max(item['score'] for item in results) < 0.5
    assert all(item['retrieval_scores']['dense'] is None for item in results)
    assert retriever.search('qzxvjkfoobar') == []


def test_batch_dense_and_bm25_are_both_fused(monkeypatch):
    monkeypatch.setenv('DERMAGUARD_RAG_ADVANCED', '0')
    retriever = LocalRagRetriever()
    calls = []
    dense_id = next(i for i, doc in enumerate(retriever.documents) if doc['document_id'] == 'GUIDE_CDC_ZOSTER')

    class FakeDense:
        def request(self, payload, timeout=None):
            calls.append(payload)
            rows = [[0.9 if index == dense_id else 0.0 for index in range(len(retriever.documents))]
                    for _ in payload['queries']]
            return {'scores': rows}

    retriever.model_client = FakeDense()
    monkeypatch.setenv('DERMAGUARD_RAG_ADVANCED', '1')
    results = retriever.search_many(['melanoma', 'basal cell carcinoma'], top_k=12)
    assert len(calls) == 1
    assert calls[0]['action'] == 'dense_queries'
    assert len(calls[0]['queries']) == 2
    assert any(item['document_id'] == 'GUIDE_CDC_ZOSTER' for item in results)
    assert any(item['retrieval_scores']['bm25'] > 0 for item in results)
    assert any(item['retrieval_scores']['dense'] == 0.9 for item in results)
    assert all(item['retrieval_scores']['rrf'] > 0 for item in results)
    assert all('RRF' in item['retrieval_backend'] for item in results)


def test_sources_and_chunk_ids_are_unique(monkeypatch):
    monkeypatch.setenv('DERMAGUARD_RAG_ADVANCED', '0')
    retriever = LocalRagRetriever()
    assert len(retriever.by_id) == len(retriever.documents)
    melanoma_pages = {doc['document_id'] for doc in retriever.documents
                      if str(doc['document_id']).startswith('GUIDE_AAD_MELANOMA')}
    assert len(melanoma_pages) == 2
    assert 'GUIDE_AAD_MELANOMA' in melanoma_pages
    results = retriever.search_many(['nevus', 'melanoma', '转诊'], top_k=12)
    assert len({item['canonical_source'] for item in results}) == len(results)
    assert any(item['source_kind'] == 'guideline_summary' and item['authoritative'] for item in results)
    assert all(not item['authoritative'] for item in results if item['source_kind'] == 'local_rule')
    assert _canonical_source('http://www.aad.org/page/?utm_source=test#x', '') == 'https://aad.org/page'
    assert _canonical_source('https://aad.org/page?id=7&utm_source=x', '') == 'https://aad.org/page?id=7'


def test_read_includes_only_same_file_neighbors(monkeypatch):
    monkeypatch.setenv('DERMAGUARD_RAG_ADVANCED', '0')
    retriever = LocalRagRetriever()
    results = retriever.read(['GUIDE_AAD_MELANOMA#C02', 'unknown'])
    assert len(results) == 3
    assert {item['id'] for item in results} == {
        'GUIDE_AAD_MELANOMA#C01', 'GUIDE_AAD_MELANOMA#C02', 'GUIDE_AAD_MELANOMA#C03',
    }
    assert [item['id'] for item in results if not item['is_neighbor']] == ['GUIDE_AAD_MELANOMA#C02']
    assert all(item['content'] and item['citation'] for item in results)
    assert len(retriever.read(['GUIDE_AAD_MELANOMA#C02'], include_neighbors=False)) == 1
    assert retriever.read(['unknown']) == []


def test_dense_worker_failure_has_cooldown(monkeypatch):
    client = RagModelClient()
    attempts = []

    def fail_start():
        attempts.append(True)
        raise OSError('worker unavailable')

    monkeypatch.setattr(client, '_start', fail_start)
    for _ in range(2):
        try:
            client.request({'action': 'prepare', 'documents': ['test']})
        except (OSError, RuntimeError):
            pass
    assert len(attempts) == 1
    assert client.process is None


def test_search_passes_remaining_budget_and_initialization_is_lazy(monkeypatch):
    monkeypatch.setenv('DERMAGUARD_RAG_ADVANCED', '1')
    calls = []

    class FakeDense:
        def request(self, payload, timeout=None):
            calls.append(timeout)
            raise RuntimeError('offline')

    monkeypatch.setattr('backend.app.services.rag.retriever._MODEL_CLIENT', FakeDense())
    retriever = LocalRagRetriever()
    assert calls == []
    results = retriever.search_many(['melanoma'], timeout=0.25)
    assert len(calls) == 1 and 0 < calls[0] <= 0.25
    assert results and '已降级' in results[0]['retrieval_backend']


def test_worker_lock_wait_respects_budget():
    client = RagModelClient()
    client.lock.acquire()
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            client.request({'action': 'prepare', 'documents': []}, timeout=0.02)
    finally:
        client.lock.release()
    assert time.monotonic() - started < 0.3


def test_partial_worker_response_is_timed_out_and_reaped(monkeypatch, tmp_path):
    # A partial line must not turn select()+readline() into an unbounded wait.
    script = tmp_path / 'partial_worker.py'
    script.write_text('import sys, time\nsys.stdin.readline()\nsys.stdout.write("{")\nsys.stdout.flush()\ntime.sleep(30)\n')
    monkeypatch.setattr('backend.app.services.rag.retriever.RAG_PYTHON', sys.executable)
    monkeypatch.setattr('backend.app.services.rag.retriever.RAG_WORKER', script)
    client = RagModelClient()
    processes = []
    original_start = client._start

    def record_start():
        original_start()
        processes.append(client.process)

    monkeypatch.setattr(client, '_start', record_start)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        client.request({'action': 'prepare', 'documents': []}, timeout=0.2)
    assert time.monotonic() - started < 1
    assert client.process is None
    assert processes[0].returncode is not None
    assert processes[0].stdin.closed and processes[0].stdout.closed


def test_interleaved_searches_keep_their_own_backend(monkeypatch):
    monkeypatch.setenv('DERMAGUARD_RAG_ADVANCED', '1')
    retriever = LocalRagRetriever()
    requests = []

    class AlternatingDense:
        def request(self, payload, timeout=None):
            requests.append(payload)
            if len(requests) > 1:
                raise RuntimeError('second request uses fallback')
            return {'scores': [[0.1] * len(retriever.documents) for _ in payload['queries']]}

    retriever.model_client = AlternatingDense()
    original_result = retriever._result
    nested_results = []
    interleaved = False

    def interleave_result(doc, backend='local_read'):
        nonlocal interleaved
        if not interleaved:
            interleaved = True
            # Simulate another request replacing shared last_backend while the
            # first request is still materializing its results.
            nested_results.extend(retriever.search('melanoma'))
        return original_result(doc, backend=backend)

    monkeypatch.setattr(retriever, '_result', interleave_result)
    outer_results = retriever.search('melanoma')
    assert len(requests) == 2
    assert nested_results and all('已降级' in item['retrieval_backend'] for item in nested_results)
    assert outer_results and all(item['retrieval_backend'] == 'BM25 + IlhaEmbed Dense / RRF' for item in outer_results)
    read_results = retriever.read([outer_results[0]['id']])
    assert read_results and all(item['retrieval_backend'] == 'local_read' for item in read_results)

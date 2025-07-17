import json
import pytest

from ai_memory.api import app

@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as client:
        yield client


def test_health(client):
    resp = client.get('/health')
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


def test_memory_and_context(client):
    resp = client.post('/memory', json={'content': 'api test mem', 'importance': 0.5})
    assert resp.status_code == 200
    mem_id = resp.get_json()['mem_id']
    assert mem_id
    resp = client.post('/context', json={'query': 'api test mem'})
    assert resp.status_code == 200
    assert 'api test mem' in resp.get_json()['context']


def test_export_json(client):
    resp = client.get('/export/json')
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, list)

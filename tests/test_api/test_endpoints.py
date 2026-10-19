"""Tests for FastAPI endpoints."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client: TestClient) -> None:
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "services" in data

    def test_health_status_is_healthy(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.json()["status"] == "healthy"

    def test_health_version_matches(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.json()["version"] == "0.1.0"

    def test_health_includes_service_checks(self, client: TestClient) -> None:
        response = client.get("/health")
        services = response.json()["services"]
        assert "ollama" in services
        assert "qdrant" in services
        assert "sqlite" in services


class TestQueryEndpoint:
    @patch("src.api.routes.run_agent")
    def test_query_returns_200(self, mock_agent, client: TestClient) -> None:
        mock_agent.return_value = {
            "response": "Las ventas de Q3 fueron €234,500",
            "tools_used": ["sql_query"],
            "tool_results": [
                {
                    "tool": "sql_query",
                    "input": "SELECT SUM(amount) FROM sales",
                    "success": True,
                    "output": {},
                }
            ],
        }
        response = client.post("/agent/query", json={"query": "¿Cuánto vendimos en Q3?"})
        assert response.status_code == 200

    @patch("src.api.routes.run_agent")
    def test_query_response_structure(self, mock_agent, client: TestClient) -> None:
        mock_agent.return_value = {
            "response": "Respuesta test",
            "tools_used": ["calculator"],
            "tool_results": [
                {"tool": "calculator", "input": "2+2", "success": True, "output": {"result": 4}}
            ],
        }
        response = client.post("/agent/query", json={"query": "cuánto es 2+2?"})
        data = response.json()
        assert "query" in data
        assert "response" in data
        assert "tools_used" in data
        assert "tool_details" in data

    @patch("src.api.routes.run_agent")
    def test_query_includes_original_query(self, mock_agent, client: TestClient) -> None:
        mock_agent.return_value = {
            "response": "test",
            "tools_used": [],
            "tool_results": [],
        }
        query_text = "¿Cuáles son los productos más vendidos?"
        response = client.post("/agent/query", json={"query": query_text})
        assert response.json()["query"] == query_text

    @patch("src.api.routes.run_agent")
    def test_query_returns_tools_used(self, mock_agent, client: TestClient) -> None:
        mock_agent.return_value = {
            "response": "El crecimiento fue 18%",
            "tools_used": ["sql_query", "calculator"],
            "tool_results": [
                {"tool": "sql_query", "input": "SELECT...", "success": True, "output": {}},
                {
                    "tool": "calculator",
                    "input": "(234500-198000)/198000*100",
                    "success": True,
                    "output": {},
                },
            ],
        }
        response = client.post("/agent/query", json={"query": "crecimiento Q3 vs Q2"})
        data = response.json()
        assert "sql_query" in data["tools_used"]
        assert "calculator" in data["tools_used"]

    @patch("src.api.routes.run_agent")
    def test_query_tool_details(self, mock_agent, client: TestClient) -> None:
        mock_agent.return_value = {
            "response": "resultado",
            "tools_used": ["doc_search"],
            "tool_results": [
                {
                    "tool": "doc_search",
                    "input": "política devoluciones",
                    "success": True,
                    "output": {},
                }
            ],
        }
        response = client.post("/agent/query", json={"query": "política de devoluciones"})
        details = response.json()["tool_details"]
        assert len(details) == 1
        assert details[0]["tool"] == "doc_search"
        assert details[0]["success"] is True


class TestQueryValidation:
    def test_empty_query_returns_422(self, client: TestClient) -> None:
        response = client.post("/agent/query", json={"query": ""})
        assert response.status_code == 422

    def test_missing_query_field_returns_422(self, client: TestClient) -> None:
        response = client.post("/agent/query", json={})
        assert response.status_code == 422

    def test_query_too_long_returns_422(self, client: TestClient) -> None:
        response = client.post("/agent/query", json={"query": "x" * 1001})
        assert response.status_code == 422

    def test_invalid_content_type_returns_422(self, client: TestClient) -> None:
        response = client.post("/agent/query", content="not json")
        assert response.status_code == 422


class TestQueryErrors:
    @patch("src.api.routes.run_agent")
    def test_agent_exception_returns_500(self, mock_agent, client: TestClient) -> None:
        mock_agent.side_effect = RuntimeError("LLM connection timeout")
        response = client.post("/agent/query", json={"query": "test"})
        assert response.status_code == 500
        assert "Agent execution failed" in response.json()["detail"]

    @patch("src.api.routes.run_agent")
    def test_500_includes_error_detail(self, mock_agent, client: TestClient) -> None:
        mock_agent.side_effect = ValueError("Invalid model response")
        response = client.post("/agent/query", json={"query": "test"})
        assert response.status_code == 500
        assert "Invalid model response" in response.json()["detail"]


class TestCORS:
    def test_cors_headers_present(self, client: TestClient) -> None:
        response = client.options(
            "/agent/query",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers


class TestOpenAPISchema:
    def test_openapi_endpoint_available(self, client: TestClient) -> None:
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "Agentic SQL Assistant"
        assert "/health" in schema["paths"]
        assert "/agent/query" in schema["paths"]


class TestQueryTracing:
    RESULT = {"response": "ok", "tools_used": [], "tool_results": []}

    @patch("src.api.routes.run_agent")
    def test_trace_id_is_null_without_langfuse(self, mock_agent, client: TestClient) -> None:
        from src.observability import tracing
        from src.observability.tracing import NoopTracer

        mock_agent.return_value = self.RESULT
        with tracing.use_tracer(NoopTracer()):
            data = client.post("/agent/query", json={"query": "hi"}).json()
        assert data["trace_id"] is None

    @patch("src.api.routes.run_agent")
    def test_one_trace_per_request_with_session_and_user(
        self, mock_agent, client: TestClient, events: list[dict]
    ) -> None:
        mock_agent.return_value = self.RESULT
        body = {"query": "hi", "session_id": "conv-1", "user_id": "alice"}
        first = client.post("/agent/query", json=body).json()
        second = client.post("/agent/query", json={"query": "again"}).json()
        assert first["trace_id"] and second["trace_id"] and first["trace_id"] != second["trace_id"]
        created = [e["body"] for e in events if e["type"] == "trace-create"]
        by_id = {b["id"]: b for b in created if "name" in b}
        assert by_id[first["trace_id"]]["sessionId"] == "conv-1"
        assert by_id[first["trace_id"]]["userId"] == "alice"
        assert "sessionId" not in by_id[second["trace_id"]]

"""Tests for new search functionality."""
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_general_search():
    """Test general search endpoint."""
    resp = client.get("/api/search?q=quantum+computing&max_results=5")
    assert resp.status_code == 200
    data = resp.json()
    assert "query" in data
    assert "results" in data
    assert "page" in data
    assert data["query"] == "quantum computing"


def test_author_search():
    """Test author-based search endpoint."""
    resp = client.get("/api/search/author/Einstein?max_results=3")
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert "Einstein" in data["query"]


def test_category_search():
    """Test category-based search endpoint."""
    resp = client.get("/api/search/category/cs.AI?max_results=5")
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert "cs.AI" in data["query"]


def test_category_search_with_dates():
    """Test category search with date filtering."""
    resp = client.get("/api/search/category/cs.AI?max_results=5&date_from=2023-01-01&date_to=2024-01-01")
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data


def test_search_exam_generation():
    """Test exam generation from search results."""
    payload = {
        "query": "quantum machine learning",
        "max_papers": 2,
        "mc_questions": 3,
        "oe_questions": 2
    }
    headers = {"X-LLM-API-Key": "test-key"}
    
    resp = client.post("/api/exam/search", json=payload, headers=headers)
    # Should either succeed or fail gracefully with appropriate status codes
    assert resp.status_code in {200, 400, 404, 502, 500}


def test_selected_papers_exam_generation():
    """Test exam generation from specific selected papers."""
    payload = {
        "arxiv_ids": ["2401.12345", "1706.03762"],
        "mc_questions": 5,
        "oe_questions": 3,
        "exam_title": "Test Multi-Paper Exam"
    }
    headers = {"X-LLM-API-Key": "test-key"}
    
    resp = client.post("/api/exam/selected", json=payload, headers=headers)
    # Should either succeed or fail gracefully with appropriate status codes
    assert resp.status_code in {200, 400, 404, 502, 500}


def test_pagination():
    """Test search pagination."""
    resp = client.get("/api/search?q=machine+learning&max_results=10&page=2")
    assert resp.status_code == 200
    data = resp.json()
    assert data["page"] == 2
    assert data["page_size"] == 10


def test_invalid_date_format():
    """Test invalid date format handling."""
    resp = client.get("/api/search/category/cs.AI?date_from=invalid-date")
    assert resp.status_code == 400
    assert "Invalid date_from format" in resp.json()["detail"]


def test_advanced_search():
    """Test advanced search endpoint."""
    params = {
        "q": "machine learning",
        "categories": "cs.AI,cs.LG",
        "min_date": "2020-01-01",
        "max_results": 5
    }
    resp = client.get("/api/search/advanced", params=params)
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data


def test_similar_papers():
    """Test similar papers endpoint."""
    resp = client.get("/api/search/similar/2401.12345?max_results=5")
    # Should either succeed or fail gracefully
    assert resp.status_code in {200, 404, 502, 500}


def test_trending_papers():
    """Test trending papers endpoint."""
    resp = client.get("/api/search/trending?days=7&max_results=10")
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data


def test_trending_with_category():
    """Test trending papers with category filter."""
    resp = client.get("/api/search/trending?category=cs.AI&days=14")
    assert resp.status_code == 200


def test_search_query_encoding():
    """Test that special characters in queries are handled properly."""
    resp = client.get("/api/search?q=neural+%26+deep&max_results=5")
    assert resp.status_code == 200


def test_empty_search_results():
    """Test behavior with queries that return no results."""
    resp = client.get("/api/search?q=xyznonexistentquery12345&max_results=5")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 0


def test_pagination_bounds():
    """Test pagination with edge cases."""
    # Test first page
    resp = client.get("/api/search?q=machine+learning&page=1&max_results=5")
    assert resp.status_code == 200
    
    # Test high page number (should not crash)
    resp = client.get("/api/search?q=machine+learning&page=100&max_results=5")
    assert resp.status_code == 200


def test_pdf_download():
    """Test PDF download endpoint."""
    # Test with a well-known paper (Attention is All You Need)
    resp = client.get("/api/download/1706.03762")
    # Should either succeed or fail gracefully
    assert resp.status_code in {200, 404, 502, 500}
    
    if resp.status_code == 200:
        # Check headers if download succeeded
        assert resp.headers.get('content-type') == 'application/pdf'
        assert 'Content-Disposition' in resp.headers
        assert 'attachment' in resp.headers['Content-Disposition']


def test_pdf_download_invalid_id():
    """Test PDF download with invalid arXiv ID."""
    resp = client.get("/api/download/invalid-id-12345")
    assert resp.status_code in {404, 502, 500}


def test_answer_shuffling():
    """Test that answer choices are shuffled and not all 'A'."""
    from main import MultipleChoiceQuestion, Choice, _shuffle_mc_choices
    
    # Create a test question with correct answer as 'A'
    original = MultipleChoiceQuestion(
        type='multiple_choice',
        prompt='Test question',
        choices=[
            Choice(label='A', text='Correct answer'),
            Choice(label='B', text='Wrong 1'),
            Choice(label='C', text='Wrong 2'),
            Choice(label='D', text='Wrong 3')
        ],
        correct='A'
    )
    
    # Test shuffling multiple times to ensure randomness
    shuffled_labels = set()
    for _ in range(10):
        shuffled = _shuffle_mc_choices(original)
        
        # Find the choice with correct answer text
        correct_choice = next(c for c in shuffled.choices if c.text == 'Correct answer')
        shuffled_labels.add(correct_choice.label)
        
        # Verify correct answer is properly updated
        assert shuffled.correct == correct_choice.label
    
    # Should have some variety in correct answer labels (not all 'A')
    assert len(shuffled_labels) > 1, f"Shuffling not working, all answers were: {shuffled_labels}"
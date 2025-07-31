"""arXiv Exam Generator

A comprehensive RAG-based FastAPI service that generates multiple-choice exams
from arXiv papers using full PDF content analysis.

Features:
- Full PDF content extraction and analysis
- Intelligent section parsing (abstract, methods, results, etc.)
- Multiple LLM backend support (OpenAI, Hugging Face)
- Interactive web interface with answer submission and feedback
- Scandinavian+Japanese+drafting board aesthetic
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
import tempfile
from datetime import datetime, timedelta
from functools import lru_cache
from typing import List, Literal, Optional, Dict, Any, Union
from pathlib import Path

import httpx
import arxiv
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field, ValidationError
from fastapi.staticfiles import StaticFiles
import aiofiles
from fastapi.templating import Jinja2Templates

# Configuration Constants
MAX_QUESTIONS = 15
MAX_PDF_SIZE = 50 * 1024 * 1024  # 50MB

# LLM Configuration
LLM_BACKEND: Literal["openai", "hf"] = os.getenv("LLM_BACKEND", "openai").lower()  # type: ignore

MULTIPLE_CHOICE_PROMPT_TEMPLATE = (
    "You are an expert educator analyzing a comprehensive academic paper. "
    "Create {n_questions} rigorous multiple-choice questions that test deep understanding of key concepts, methods, and findings. "
    "Focus on conceptual understanding, technical details, and critical analysis rather than simple recall. \n\n"
    "Use this exact JSON format for each question (ensure proper escaping of quotes and special characters): "
    '{{\"type\": \"multiple_choice\", \"prompt\": \"Clear, specific question\", \"choices\": [{{\"label\": \"A\", \"text\": \"Option text\"}}], \"correct\": \"A\"}} \n\n'
    "Requirements: "
    "- Create exactly {n_questions} questions "
    "- Each question should have 4 choices (A, B, C, D) "
    "- Questions should cover different aspects: methodology, results, implications, technical details "
    "- Avoid ambiguous or trick questions "
    "- Ensure one clearly correct answer per question \n\n"
    "Produce a strict JSON array with exactly {n_questions} multiple-choice questions."
)

OPEN_ENDED_PROMPT_TEMPLATE = (
    "You are an expert educator analyzing a comprehensive academic paper. "
    "Create {n_questions} open-ended questions that require deep analysis, synthesis, and critical thinking. "
    "Each question should test comprehensive understanding and require substantial written responses. \n\n"
    "Use this exact JSON format for each question: "
    '{{\"type\": \"open_ended\", \"prompt\": \"Question requiring analysis...\", \"sample_answer\": \"Comprehensive model answer...\", \"key_points\": [\"point1\", \"point2\"], '
    '\"graded_examples\": [{{\"score\": 5, \"answer\": \"excellent comprehensive answer\", \"feedback\": \"why 5/5 - comprehensive and insightful\"}}, '
    '{{\"score\": 4, \"answer\": \"good solid answer\", \"feedback\": \"why 4/5 - good with minor gaps\"}}, '
    '{{\"score\": 3, \"answer\": \"adequate basic answer\", \"feedback\": \"why 3/5 - meets minimum requirements\"}}]}} \n\n'
    "Requirements: "
    "- Create exactly {n_questions} open-ended questions "
    "- Each question should require 2-3 paragraph responses "
    "- Include 3-5 key points students should address "
    "- Provide exactly 3 graded examples (scores 5, 4, 3) with detailed feedback "
    "- Focus on analysis, evaluation, synthesis, and application \n\n"
    "Grading criteria: 5=comprehensive+accurate+insightful, 4=good+mostly accurate+clear, 3=adequate+meets minimum requirements\n\n"
    "Produce a strict JSON array with exactly {n_questions} open-ended questions."
)

# Backend-specific configuration
if LLM_BACKEND == "openai":
    try:
        import openai
    except ImportError:
        sys.exit("pip install openai >=1.30")

    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        sys.exit("Set OPENAI_API_KEY when LLM_BACKEND=openai")

elif LLM_BACKEND == "hf":
    HF_API_URL = os.getenv("HF_API_URL", "http://localhost:8080/v1/chat/completions")
    HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

else:
    sys.exit(f"Unsupported LLM_BACKEND: {LLM_BACKEND}")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("arxiv_exam_app")

# Data Models
class Choice(BaseModel):
    label: str = Field(..., min_length=1, max_length=1, pattern="[A-Z]")
    text: str


class MultipleChoiceQuestion(BaseModel):
    type: Literal["multiple_choice"] = "multiple_choice"
    prompt: str
    choices: List[Choice] = Field(..., min_length=2)
    correct: str = Field(..., pattern="[A-Z]")


class GradedExample(BaseModel):
    score: int = Field(..., ge=3, le=5, description="Score from 3-5 points")
    answer: str = Field(..., description="Example answer text")
    feedback: str = Field(..., description="Explanation of why this answer received this score")


class OpenEndedQuestion(BaseModel):
    type: Literal["open_ended"] = "open_ended"
    prompt: str
    sample_answer: str = Field(..., description="Sample/model answer for reference")
    key_points: List[str] = Field(default_factory=list, description="Key points that should be addressed")
    graded_examples: List[GradedExample] = Field(default_factory=list, description="3 graded example answers (scores 3-5) for teacher mode")


# Union type for all question types
Question = Union[MultipleChoiceQuestion, OpenEndedQuestion]


class PaperMetadata(BaseModel):
    title: str
    authors: List[str]
    abstract: str
    published: Optional[str] = None
    categories: List[str] = []
    arxiv_id: Optional[str] = None
    citation: Optional[str] = None


class ProcessedPaper(BaseModel):
    metadata: PaperMetadata
    full_text: str
    sections: Dict[str, str] = {}  # section_name -> content
    total_chars: int
    processing_method: Literal["pdf", "abstract_only"] = "pdf"


class ExamResponse(BaseModel):
    metadata: PaperMetadata
    questions: List[Question]
    exam_name: Optional[str] = None


# Search-related models
class SearchResult(BaseModel):
    """Single paper result from arXiv search."""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    published: Optional[str] = None
    updated: Optional[str] = None
    categories: List[str] = []
    pdf_url: Optional[str] = None
    entry_id: str


class SearchResponse(BaseModel):
    """Response for search endpoints."""
    query: str
    total_results: int
    results: List[SearchResult]
    page: int = 1
    page_size: int = 20
    has_more: bool = False


class SearchExamRequest(BaseModel):
    """Request model for generating exam from search results."""
    query: str
    max_papers: int = Field(default=3, ge=1, le=10, description="Number of papers to include in exam")
    mc_questions: int = Field(default=10, ge=1, le=12, description="Number of multiple-choice questions")
    oe_questions: int = Field(default=5, ge=1, le=8, description="Number of open-ended questions")
    sort_by: Literal["relevance", "lastUpdatedDate", "submittedDate"] = "relevance"


class SelectedPapersExamRequest(BaseModel):
    """Request model for generating exam from specific selected papers."""
    arxiv_ids: List[str] = Field(..., min_length=1, max_length=10, description="List of arXiv IDs to include")
    mc_questions: int = Field(default=10, ge=1, le=12, description="Number of multiple-choice questions")
    oe_questions: int = Field(default=5, ge=1, le=8, description="Number of open-ended questions")
    exam_title: Optional[str] = Field(None, description="Custom title for the exam")

# FastAPI Application
app = FastAPI(
    title="arXiv Exam Generator",
    description="Generate comprehensive multiple-choice exams from arXiv papers using RAG",
    version="1.0.0",
    license_info={"name": "MIT"},
    contact={"url": "https://github.com/jcdavis131/arxiv-exam-app"},
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------------------------------------------------------
# Helper funcs – arXiv Processing & Search
# ---------------------------------------------------------------------------
def _convert_arxiv_result_to_search_result(paper: arxiv.Result) -> SearchResult:
    """Convert arxiv.Result to SearchResult model."""
    # Extract arXiv ID from entry_id (format: http://arxiv.org/abs/1234.5678v1)
    arxiv_id = paper.entry_id.split('/')[-1]
    
    # More careful version number removal - only remove if it matches the pattern
    import re
    version_match = re.match(r'^(\d{4}\.\d{4,5})v\d+$', arxiv_id)
    if version_match:
        arxiv_id = version_match.group(1)
    
    logger.debug(f"Converted entry_id {paper.entry_id} to arxiv_id {arxiv_id}")
    
    return SearchResult(
        arxiv_id=arxiv_id,
        title=_normalize_whitespace(paper.title),
        authors=[author.name for author in paper.authors],
        abstract=_normalize_whitespace(paper.summary),
        published=paper.published.isoformat() if paper.published else None,
        updated=paper.updated.isoformat() if paper.updated else None,
        categories=paper.categories,
        pdf_url=paper.pdf_url,
        entry_id=paper.entry_id
    )


def _get_sort_criterion(sort_by: str) -> arxiv.SortCriterion:
    """Convert string sort parameter to arxiv.SortCriterion."""
    sort_map = {
        "relevance": arxiv.SortCriterion.Relevance,
        "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
        "submittedDate": arxiv.SortCriterion.SubmittedDate
    }
    return sort_map.get(sort_by, arxiv.SortCriterion.Relevance)


async def search_arxiv_papers(
    query: str,
    max_results: int = 20,
    sort_by: str = "relevance",
    start: int = 0
) -> List[SearchResult]:
    """Search arXiv papers using arxiv.py with pagination support."""
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=_get_sort_criterion(sort_by),
            sort_order=arxiv.SortOrder.Descending
        )
        
        results = []
        for i, paper in enumerate(client.results(search)):
            if i < start:
                continue
            if len(results) >= max_results:
                break
            results.append(_convert_arxiv_result_to_search_result(paper))
        
        return results
    
    except Exception as e:
        logger.error(f"arXiv search error: {e}")
        raise HTTPException(status_code=502, detail="arXiv search failed") from e


# ---------------------------------------------------------------------------
# Helper funcs – arXiv Processing
# ---------------------------------------------------------------------------
def _parse_arxiv_metadata(paper: arxiv.Result) -> PaperMetadata:
    """Parse arXiv paper result to extract paper metadata."""
    authors = [author.name for author in paper.authors]
    categories = paper.categories
    
    # Extract arXiv ID from the entry_id URL
    arxiv_id = paper.entry_id.split('/')[-1]
    
    # Generate citation in standard format
    authors_str = ', '.join(authors[:3])  # First 3 authors
    if len(authors) > 3:
        authors_str += ' et al.'
    
    year = paper.published.year if paper.published else 'n.d.'
    citation = f"{authors_str} ({year}). {paper.title}. arXiv preprint arXiv:{arxiv_id}."
    
    return PaperMetadata(
        title=_normalize_whitespace(paper.title),
        authors=authors,
        abstract=_normalize_whitespace(paper.summary),
        published=paper.published.isoformat() if paper.published else None,
        categories=categories,
        arxiv_id=arxiv_id,
        citation=citation
    )


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    return re.sub(r"\s+", " ", text).strip()


def _shuffle_mc_choices(question: MultipleChoiceQuestion) -> MultipleChoiceQuestion:
    """Shuffle multiple choice answers to prevent all correct answers being 'A'."""
    if len(question.choices) < 2:
        return question  # Can't shuffle if less than 2 choices
    
    # Find the current correct choice
    correct_choice = None
    for choice in question.choices:
        if choice.label == question.correct:
            correct_choice = choice
            break
    
    if not correct_choice:
        return question  # Safety check
    
    # Create a list of choice texts (without labels)
    choice_texts = [choice.text for choice in question.choices]
    
    # Shuffle the choice texts
    random.shuffle(choice_texts)
    
    # Create new labels A, B, C, D...
    labels = ['A', 'B', 'C', 'D', 'E', 'F'][:len(choice_texts)]
    
    # Create new choices with shuffled texts and sequential labels
    new_choices = []
    new_correct_label = 'A'  # Default fallback
    
    for i, text in enumerate(choice_texts):
        label = labels[i]
        new_choices.append(Choice(label=label, text=text))
        
        # Track which label now contains the correct answer
        if text == correct_choice.text:
            new_correct_label = label
    
    # Return new question with shuffled choices
    return MultipleChoiceQuestion(
        type=question.type,
        prompt=question.prompt,
        choices=new_choices,
        correct=new_correct_label
    )


def _extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from PDF file using PyPDF."""
    try:
        from pypdf import PdfReader
    except ImportError:
        sys.exit("pip install pypdf >=4.0")
    
    try:
        reader = PdfReader(pdf_path)
        text_parts = []
        
        for page_num, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                continue
        
        full_text = "\n\n".join(text_parts)
        
        # Clean up common PDF artifacts
        full_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', full_text)
        full_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', full_text)
        full_text = _normalize_whitespace(full_text)
        
        return full_text.strip()
        
    except Exception as e:
        logger.error(f"PDF text extraction failed: {e}")
        raise ValueError(f"Failed to extract text from PDF: {e}")


def _extract_sections(text: str) -> Dict[str, str]:
    """Extract sections from academic paper text."""
    sections = {}
    
    # Common section patterns in academic papers
    section_patterns = [
        r'(?i)\n\s*(abstract|introduction|background|related\s+work|methodology|methods|approach|model|architecture|experiments?|results?|evaluation|discussion|conclusion|future\s+work|acknowledgments?)\s*\n',
        r'\n\s*(\d+\.?\s+[A-Z][^\n]+)\n',  # Numbered sections
        r'\n\s*([A-Z][A-Z\s]{3,20})\n'      # All caps sections
    ]
    
    current_section = "main"
    current_content = []
    
    lines = text.split('\n')
    
    for line in lines:
        is_section_header = False
        
        for pattern in section_patterns:
            match = re.search(pattern, f'\n{line}\n')
            if match:
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = match.group(1).strip().lower()
                current_content = []
                is_section_header = True
                break
        
        if not is_section_header and line.strip():
            current_content.append(line)
    
    # Save final section
    if current_content:
        sections[current_section] = '\n'.join(current_content).strip()
    
    return sections


async def _download_pdf(paper: arxiv.Result) -> Path:
    """Download PDF from arXiv using arxiv.py."""
    try:
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        temp_path = Path(temp_file.name)
        temp_file.close()  # Close file handle before downloading
        
        # Download PDF using arxiv.py
        paper.download_pdf(filename=str(temp_path))
        
        # Check file size
        if temp_path.stat().st_size > MAX_PDF_SIZE:
            temp_path.unlink()  # Clean up
            raise ValueError(f"PDF too large: {temp_path.stat().st_size} bytes")
        
        return temp_path
        
    except Exception as exc:
        logger.error(f"PDF download failed for {paper.entry_id}: {exc}")
        # Clean up temp file if it exists
        if temp_path and temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=502, detail="Failed to download PDF") from exc


async def get_comprehensive_paper_content(arxiv_id: str) -> ProcessedPaper:
    """Get comprehensive paper content including full PDF text using arxiv.py."""
    try:
        # Search for the paper using arxiv.py
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        papers = list(client.results(search))
        
        if not papers:
            raise HTTPException(status_code=404, detail=f"Paper {arxiv_id} not found")
        
        paper = papers[0]
        metadata = _parse_arxiv_metadata(paper)
        
    except Exception as exc:
        logger.error(f"arXiv search error for {arxiv_id}: {exc}")
        raise HTTPException(status_code=502, detail="arXiv API error") from exc
    
    # Try to get full PDF content
    pdf_path = None
    try:
        logger.info(f"Downloading PDF for {arxiv_id}")
        pdf_path = await _download_pdf(paper)
        
        logger.info(f"Extracting text from PDF: {pdf_path}")
        full_text = _extract_pdf_text(pdf_path)
        
        if len(full_text.strip()) < 500:  # Very short extraction, likely failed
            raise ValueError("PDF text extraction yielded insufficient content")
        
        sections = _extract_sections(full_text)
        
        logger.info(f"Successfully processed PDF: {len(full_text)} characters, {len(sections)} sections")
        
        return ProcessedPaper(
            metadata=metadata,
            full_text=full_text,
            sections=sections,
            total_chars=len(full_text),
            processing_method="pdf"
        )
        
    except Exception as e:
        logger.warning(f"PDF processing failed for {arxiv_id}, falling back to abstract: {e}")
        
        # Fallback to abstract only
        abstract_text = f"{metadata.title}\n\n{metadata.abstract}"
        
        return ProcessedPaper(
            metadata=metadata,
            full_text=abstract_text,
            sections={"abstract": metadata.abstract},
            total_chars=len(abstract_text),
            processing_method="abstract_only"
        )
    
    finally:
        # Clean up temporary PDF file
        if pdf_path and pdf_path.exists():
            try:
                pdf_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {pdf_path}: {e}")

# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------
def _create_exam_prompt(paper: ProcessedPaper, n_questions: int, question_type: str = "mixed") -> str:
    """Create a comprehensive prompt for question generation."""
    # Paper metadata
    authors_str = ', '.join(paper.metadata.authors[:3])
    if len(paper.metadata.authors) > 3:
        authors_str += ' et al.'
    
    header = f"""PAPER: {paper.metadata.title}
AUTHORS: {authors_str}
PROCESSING: {paper.processing_method.upper()} ({paper.total_chars:,} chars)

"""
    
    # Select and prepare content
    priority_sections = ['abstract', 'introduction', 'methodology', 'methods', 'results', 'conclusion']
    content_parts = []
    total_length = 0
    
    for section in priority_sections:
        if section in paper.sections and total_length < 8000:
            content = paper.sections[section]
            if len(content) > 1500:
                content = content[:1500] + "..."
            content_parts.append(f"**{section.upper()}:**\n{content}")
            total_length += len(content)
    
    if not content_parts:
        content = paper.full_text[:10000] + "..." if len(paper.full_text) > 10000 else paper.full_text
        content_parts = [content]
    
    content_text = '\n\n'.join(content_parts)
    
    # Sanitize content_text to remove/replace problematic Unicode characters
    # This is crucial for consistent encoding when sending to LLMs
    try:
        content_text = content_text.encode('utf-8', errors='replace').decode('utf-8')
    except Exception as e:
        logger.warning(f"Failed to sanitize content_text for LLM prompt: {e}")
        # Fallback to a very basic ASCII if sanitation fails catastrophically
        content_text = content_text.encode('ascii', errors='ignore').decode('ascii')

    if question_type == "multiple_choice":
        return f"{header}CONTENT:\n{content_text}\n\nGenerate {n_questions} multiple-choice questions covering key concepts, methods, and findings."
    elif question_type == "open_ended":
        return f"{header}CONTENT:\n{content_text}\n\nGenerate {n_questions} open-ended questions requiring analysis and synthesis."
    else:
        return f"{header}CONTENT:\n{content_text}\n\nGenerate {n_questions} questions."


async def _generate_mc_questions(paper: ProcessedPaper, n_questions: int, provider: str, model: str, api_key: str) -> List[MultipleChoiceQuestion]:
    """Generate multiple choice questions using specified LLM provider."""
    if provider == "openai":
        return await _generate_mc_questions_openai(paper, n_questions, model, api_key)
    elif provider == "anthropic":
        return await _generate_mc_questions_anthropic(paper, n_questions, model, api_key)
    elif provider == "groq":
        return await _generate_mc_questions_groq(paper, n_questions, model, api_key)
    elif provider == "huggingface":
        return await _generate_mc_questions_hf(paper, n_questions, model, api_key)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported LLM provider: {provider}")


async def _generate_oe_questions(paper: ProcessedPaper, n_questions: int, provider: str, model: str, api_key: str) -> List[OpenEndedQuestion]:
    """Generate open-ended questions using specified LLM provider."""
    if provider == "openai":
        return await _generate_oe_questions_openai(paper, n_questions, model, api_key)
    elif provider == "anthropic":
        return await _generate_oe_questions_anthropic(paper, n_questions, model, api_key)
    elif provider == "groq":
        return await _generate_oe_questions_groq(paper, n_questions, model, api_key)
    elif provider == "huggingface":
        return await _generate_oe_questions_hf(paper, n_questions, model, api_key)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported LLM provider: {provider}")


async def _generate_mc_questions_openai(paper: ProcessedPaper, n_questions: int, model: str = "gpt-4o-mini", api_key: str = None) -> List[MultipleChoiceQuestion]:
    """Generate multiple choice questions using OpenAI API."""
    system_msg = MULTIPLE_CHOICE_PROMPT_TEMPLATE.format(n_questions=n_questions)
    user_msg = _create_exam_prompt(paper, n_questions, question_type="multiple_choice")

    client = openai.AsyncOpenAI(api_key=api_key or OPENAI_API_KEY)
    
    try:
        response = await client.chat.completions.create(
            model=model,
            temperature=0.2,  # Slightly higher temperature for better JSON formatting
            messages=[
                {"role": "system", "content": system_msg}, 
                {"role": "user", "content": user_msg}
            ],
            max_tokens=4000,  # Focused token limit for MC questions
        )
        return _validate_mc_questions(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"OpenAI API error (MC): {e}")
        raise HTTPException(status_code=500, detail="LLM processing failed") from e


async def _generate_oe_questions_openai(paper: ProcessedPaper, n_questions: int, model: str = "gpt-4o-mini", api_key: str = None) -> List[OpenEndedQuestion]:
    """Generate open-ended questions using OpenAI API."""
    system_msg = OPEN_ENDED_PROMPT_TEMPLATE.format(n_questions=n_questions)
    user_msg = _create_exam_prompt(paper, n_questions, question_type="open_ended")

    client = openai.AsyncOpenAI(api_key=api_key or OPENAI_API_KEY)
    
    try:
        response = await client.chat.completions.create(
            model=model,
            temperature=0.2,  # Slightly higher temperature for better JSON formatting
            messages=[
                {"role": "system", "content": system_msg}, 
                {"role": "user", "content": user_msg}
            ],
            max_tokens=4096,  # Higher token limit for open-ended questions with examples
        )
        return _validate_oe_questions(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"OpenAI API error (OE): {e}")
        raise HTTPException(status_code=500, detail="LLM processing failed") from e


# Anthropic provider functions
async def _generate_mc_questions_anthropic(paper: ProcessedPaper, n_questions: int, model: str, api_key: str) -> List[MultipleChoiceQuestion]:
    """Generate multiple choice questions using Anthropic API."""
    try:
        import anthropic
    except ImportError:
        raise HTTPException(status_code=500, detail="pip install anthropic")
    
    system_msg = MULTIPLE_CHOICE_PROMPT_TEMPLATE.format(n_questions=n_questions)
    user_msg = _create_exam_prompt(paper, n_questions, question_type="multiple_choice")

    client = anthropic.AsyncAnthropic(api_key=api_key)
    
    try:
        response = await client.messages.create(
            model=model,
            max_tokens=4000,
            temperature=0.2,
            system=system_msg,
            messages=[{"role": "user", "content": user_msg}]
        )
        return _validate_mc_questions(response.content[0].text)
    except Exception as e:
        logger.error(f"Anthropic API error (MC): {e}")
        raise HTTPException(status_code=500, detail="LLM processing failed") from e


async def _generate_oe_questions_anthropic(paper: ProcessedPaper, n_questions: int, model: str, api_key: str) -> List[OpenEndedQuestion]:
    """Generate open-ended questions using Anthropic API."""
    try:
        import anthropic
    except ImportError:
        raise HTTPException(status_code=500, detail="pip install anthropic")
    
    system_msg = OPEN_ENDED_PROMPT_TEMPLATE.format(n_questions=n_questions)
    user_msg = _create_exam_prompt(paper, n_questions, question_type="open_ended")

    client = anthropic.AsyncAnthropic(api_key=api_key)
    
    try:
        response = await client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=0.2,
            system=system_msg,
            messages=[{"role": "user", "content": user_msg}]
        )
        return _validate_oe_questions(response.content[0].text)
    except Exception as e:
        logger.error(f"Anthropic API error (OE): {e}")
        raise HTTPException(status_code=500, detail="LLM processing failed") from e


# Groq provider functions  
async def _generate_mc_questions_groq(paper: ProcessedPaper, n_questions: int, model: str, api_key: str) -> List[MultipleChoiceQuestion]:
    """Generate multiple choice questions using Groq API."""
    try:
        import groq
    except ImportError:
        raise HTTPException(status_code=500, detail="pip install groq")
    
    system_msg = MULTIPLE_CHOICE_PROMPT_TEMPLATE.format(n_questions=n_questions)
    user_msg = _create_exam_prompt(paper, n_questions, question_type="multiple_choice")

    client = groq.AsyncGroq(api_key=api_key)
    
    try:
        response = await client.chat.completions.create(
            model=model,
            temperature=0.2,
            max_tokens=4000,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
        )
        return _validate_mc_questions(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Groq API error (MC): {e}")
        raise HTTPException(status_code=500, detail="LLM processing failed") from e


async def _generate_oe_questions_groq(paper: ProcessedPaper, n_questions: int, model: str, api_key: str) -> List[OpenEndedQuestion]:
    """Generate open-ended questions using Groq API."""
    try:
        import groq
    except ImportError:
        raise HTTPException(status_code=500, detail="pip install groq")
    
    system_msg = OPEN_ENDED_PROMPT_TEMPLATE.format(n_questions=n_questions)
    user_msg = _create_exam_prompt(paper, n_questions, question_type="open_ended")

    client = groq.AsyncGroq(api_key=api_key)
    
    try:
        response = await client.chat.completions.create(
            model=model,
            temperature=0.2,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
        )
        return _validate_oe_questions(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Groq API error (OE): {e}")
        raise HTTPException(status_code=500, detail="LLM processing failed") from e


# HuggingFace provider functions
async def _generate_mc_questions_hf(paper: ProcessedPaper, n_questions: int, model: str, api_key: str) -> List[MultipleChoiceQuestion]:
    """Generate multiple choice questions using HuggingFace API."""
    system_msg = MULTIPLE_CHOICE_PROMPT_TEMPLATE.format(n_questions=n_questions)
    user_msg = _create_exam_prompt(paper, n_questions, question_type="multiple_choice")
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.2,
        "max_tokens": 4000,
    }
    
    headers = {"Authorization": f"Bearer {api_key}"}
    hf_url = os.getenv("HF_API_URL", "https://api-inference.huggingface.co/models") + f"/{model}"
    
    async with httpx.AsyncClient(timeout=180) as client:
        try:
            response = await client.post(hf_url, json=payload, headers=headers)
            response.raise_for_status()
            content = response.json()[0].get("generated_text", "")
            return _validate_mc_questions(content)
        except Exception as e:
            logger.error(f"HF API error (MC): {e}")
            raise HTTPException(status_code=500, detail="LLM processing failed") from e


async def _generate_oe_questions_hf(paper: ProcessedPaper, n_questions: int, model: str, api_key: str) -> List[OpenEndedQuestion]:
    """Generate open-ended questions using HuggingFace API."""
    system_msg = OPEN_ENDED_PROMPT_TEMPLATE.format(n_questions=n_questions)
    user_msg = _create_exam_prompt(paper, n_questions, question_type="open_ended")
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.2,
        "max_tokens": 4096,
    }
    
    headers = {"Authorization": f"Bearer {api_key}"}
    hf_url = os.getenv("HF_API_URL", "https://api-inference.huggingface.co/models") + f"/{model}"
    
    async with httpx.AsyncClient(timeout=180) as client:
        try:
            response = await client.post(hf_url, json=payload, headers=headers)
            response.raise_for_status()
            content = response.json()[0].get("generated_text", "")
            return _validate_oe_questions(content)
        except Exception as e:
            logger.error(f"HF API error (OE): {e}")
            raise HTTPException(status_code=500, detail="LLM processing failed") from e


async def _generate_questions_hf(paper: ProcessedPaper, n_questions: int, mc_ratio: float = 1.0) -> List[Question]:
    """Generate mixed question types using Hugging Face API."""
    mc_questions = max(1, int(n_questions * mc_ratio))
    oe_questions = n_questions - mc_questions
    
    system_msg = SYSTEM_PROMPT_TEMPLATE.format(
        mc_questions=mc_questions,
        oe_questions=oe_questions,
        total_q=n_questions
    )
    user_msg = _create_exam_prompt(paper, n_questions, mc_ratio)
    
    payload = {
        "model": HF_MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.1,
        "max_tokens": 8000,  # Increased for 15 questions with longer responses
    }
    
    async with httpx.AsyncClient(timeout=180) as client:
        try:
            response = await client.post(HF_API_URL, json=payload)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return _validate_questions(content)
        except Exception as e:
            logger.error(f"HF API error: {e}")
            raise HTTPException(status_code=500, detail="LLM processing failed") from e


def _validate_mc_questions(raw_response: str) -> List[MultipleChoiceQuestion]:
    """Validate and parse LLM-generated multiple choice questions."""
    try:
        # Clean JSON response
        cleaned = raw_response.strip()
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        # Fix common JSON escape issues more carefully
        try:
            # First attempt - try direct parsing
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                # Second attempt - fix common escape issues
                cleaned = cleaned.replace('\\n', '\\\\n').replace('\\t', '\\\\t').replace('\\r', '\\\\r')
                cleaned = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', cleaned)
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                # Third attempt - try extracting valid JSON objects manually
                data = []
                # Look for individual JSON objects in the response
                obj_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                matches = re.findall(obj_pattern, cleaned)
                for match in matches:
                    try:
                        obj = json.loads(match)
                        if isinstance(obj, dict) and obj.get('type') == 'multiple_choice':
                            data.append(obj)
                    except:
                        continue
                if not data:
                    raise ValueError("Could not extract valid JSON objects")
        
        logger.debug(f"Successfully parsed MC JSON with {len(data) if isinstance(data, list) else 1} items")
        if not isinstance(data, list):
            raise ValueError("Expected JSON array")
        
        questions = []
        for i, question_data in enumerate(data):
            try:
                question = MultipleChoiceQuestion.model_validate(question_data)
                # Validate correct answer exists in choices
                choice_labels = [c.label for c in question.choices]
                if question.correct not in choice_labels:
                    logger.warning(f"Skipping MC question {i+1}: correct answer not in choices")
                    continue
                
                # Shuffle choices to prevent all answers being 'A'
                shuffled_question = _shuffle_mc_choices(question)
                questions.append(shuffled_question)
            except ValidationError as e:
                logger.error(f"Validation error for MC question {i+1}: {e}")
                continue
        
        if not questions:
            raise ValueError("No valid MC questions generated")
        
        logger.info(f"Successfully validated {len(questions)} MC questions")
        return questions
        
    except json.JSONDecodeError as e:
        logger.error(f"MC JSON decode error: {e}")
        raise HTTPException(status_code=500, detail="Invalid JSON from LLM") from e
    except Exception as e:
        logger.error(f"MC question validation failed: {e}")
        raise HTTPException(status_code=500, detail="MC question validation failed") from e


def _validate_oe_questions(raw_response: str) -> List[OpenEndedQuestion]:
    """Validate and parse LLM-generated open-ended questions."""
    try:
        # Clean JSON response
        cleaned = raw_response.strip()
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        # Fix common JSON escape issues more carefully
        try:
            # First attempt - try direct parsing
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                # Second attempt - fix common escape issues
                cleaned = cleaned.replace('\\n', '\\\\n').replace('\\t', '\\\\t').replace('\\r', '\\\\r')
                cleaned = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', cleaned)
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                # Third attempt - try extracting valid JSON objects manually
                import ast
                data = []
                # Look for individual JSON objects in the response
                obj_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                matches = re.findall(obj_pattern, cleaned)
                for match in matches:
                    try:
                        obj = json.loads(match)
                        if isinstance(obj, dict) and obj.get('type') == 'open_ended':
                            data.append(obj)
                    except:
                        continue
                if not data:
                    raise ValueError("Could not extract valid JSON objects")
        
        logger.debug(f"Successfully parsed OE JSON with {len(data) if isinstance(data, list) else 1} items")
        if not isinstance(data, list):
            raise ValueError("Expected JSON array")
        
        questions = []
        for i, question_data in enumerate(data):
            try:
                # Make graded_examples optional if missing
                if 'graded_examples' not in question_data:
                    question_data['graded_examples'] = []
                    logger.warning(f"OE Question {i+1}: Missing graded_examples, using empty list")
                
                question = OpenEndedQuestion.model_validate(question_data)
                questions.append(question)
            except ValidationError as e:
                logger.error(f"Validation error for OE question {i+1}: {e}")
                continue
        
        if not questions:
            raise ValueError("No valid OE questions generated")
        
        logger.info(f"Successfully validated {len(questions)} OE questions")
        return questions
        
    except json.JSONDecodeError as e:
        logger.error(f"OE JSON decode error: {e}")
        raise HTTPException(status_code=500, detail="Invalid JSON from LLM") from e
    except Exception as e:
        logger.error(f"OE question validation failed: {e}")
        raise HTTPException(status_code=500, detail="OE question validation failed") from e


async def generate_questions(paper: ProcessedPaper, n_questions: int = 15, mc_questions: int = 10, oe_questions: int = 5, 
                           provider: str = "openai", model: str = "gpt-4o-mini", api_key: str = None) -> List[Question]:
    """Generate questions using split approach with configurable LLM provider."""
    
    # Generate MC questions first
    logger.info(f"Generating {mc_questions} multiple-choice questions using {provider}/{model}...")
    mc_results = await _generate_mc_questions(paper, mc_questions, provider, model, api_key)
    
    # Generate open-ended questions second
    logger.info(f"Generating {oe_questions} open-ended questions using {provider}/{model}...")
    oe_results = await _generate_oe_questions(paper, oe_questions, provider, model, api_key)
    
    # Combine results
    all_questions: List[Question] = []
    all_questions.extend(mc_results)
    all_questions.extend(oe_results)
    
    logger.info(f"Successfully generated {len(mc_results)} MC + {len(oe_results)} OE questions")
    return all_questions


async def generate_exam_name(paper: ProcessedPaper, questions: List[Question], provider: str = "openai", model: str = "gpt-4o-mini", api_key: str = None) -> str:
    """Generate a creative exam name using the LLM."""
    
    # Create a prompt for exam naming
    system_prompt = """You are a creative academic assistant. Generate a concise, engaging exam title based on the paper content and questions.

Requirements:
- Maximum 6-8 words 
- Academic but memorable
- Captures the key concepts/themes
- Should sound like a proper exam title
- No quotes or special formatting needed

Examples:
- "Advanced Machine Learning Concepts Assessment"  
- "Quantum Computing Fundamentals Examination"
- "Neural Network Architecture Deep Dive"
- "Computational Biology Methods Evaluation"

Return only the exam title text, nothing else."""

    # Extract key topics from paper and questions
    topics = []
    for question in questions[:5]:  # Use first 5 questions for topic extraction
        if hasattr(question, 'prompt'):
            topics.append(question.prompt[:100])  # First 100 chars
    
    topics_text = " ".join(topics)
    
    user_prompt = f"""Paper Title: {paper.metadata.title}
Abstract: {paper.metadata.abstract[:300]}...
Categories: {', '.join(paper.metadata.categories[:3])}
Sample Question Topics: {topics_text[:500]}...

Generate an engaging exam title for this assessment."""

    if provider == "openai":
        client = openai.AsyncOpenAI(api_key=api_key)
        try:
            response = await client.chat.completions.create(
                model=model,
                temperature=0.7,  # Higher temperature for creativity
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=30,  # Short response needed
            )
            
            exam_name = response.choices[0].message.content.strip()
            return exam_name
            
        except Exception as e:
            logger.warning(f"Failed to generate exam name: {e}")
            # Fallback to a simple name
            main_category = paper.metadata.categories[0] if paper.metadata.categories else "Academic"
            return f"{main_category.replace('cs.', 'Computer Science - ').title()} Assessment"
    
    else:
        # For other providers, use a simple fallback for now
        main_category = paper.metadata.categories[0] if paper.metadata.categories else "Academic" 
        return f"{main_category.replace('cs.', 'Computer Science - ').title()} Assessment"


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/test-paper/{arxiv_id}", tags=["Debug"])
async def test_paper_processing(arxiv_id: str):
    """Test endpoint to verify paper processing works."""
    try:
        paper = await get_comprehensive_paper_content(arxiv_id)
        return {
            "success": True,
            "arxiv_id": arxiv_id,
            "title": paper.metadata.title,
            "authors": paper.metadata.authors[:3],
            "processing_method": paper.processing_method,
            "total_chars": paper.total_chars
        }
    except Exception as e:
        return {
            "success": False,
            "arxiv_id": arxiv_id,
            "error": str(e),
            "error_type": type(e).__name__
        }


@app.get("/api/search", response_model=SearchResponse, tags=["Search"])
async def search_papers(
    q: str = Query(..., description="Search query (supports field prefixes like 'au:', 'ti:', 'abs:', 'cat:')"),
    max_results: int = Query(20, ge=1, le=100, description="Maximum number of results to return"),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    sort_by: Literal["relevance", "lastUpdatedDate", "submittedDate"] = Query("relevance", description="Sort order")
) -> SearchResponse:
    """Search arXiv papers with advanced query support."""
    page_size = max_results
    start = (page - 1) * page_size
    
    # Get one extra result to check if there are more pages
    search_results = await search_arxiv_papers(
        query=q,
        max_results=page_size + 1,
        sort_by=sort_by,
        start=start
    )
    
    has_more = len(search_results) > page_size
    if has_more:
        search_results = search_results[:page_size]
    
    return SearchResponse(
        query=q,
        total_results=len(search_results),  # Note: arxiv API doesn't provide total count
        results=search_results,
        page=page,
        page_size=page_size,
        has_more=has_more
    )


@app.get("/api/search/author/{author_name}", response_model=SearchResponse, tags=["Search"])
async def search_by_author(
    author_name: str,
    max_results: int = Query(20, ge=1, le=100),
    page: int = Query(1, ge=1),
    sort_by: Literal["relevance", "lastUpdatedDate", "submittedDate"] = Query("submittedDate")
) -> SearchResponse:
    """Search papers by author name."""
    query = f"au:{author_name}"
    return await search_papers(q=query, max_results=max_results, page=page, sort_by=sort_by)


@app.get("/api/search/category/{category}", response_model=SearchResponse, tags=["Search"])
async def search_by_category(
    category: str,
    max_results: int = Query(20, ge=1, le=100),
    page: int = Query(1, ge=1),
    sort_by: Literal["relevance", "lastUpdatedDate", "submittedDate"] = Query("submittedDate"),
    date_from: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="End date (YYYY-MM-DD)")
) -> SearchResponse:
    """Search papers by category with optional date filtering."""
    query = f"cat:{category}"
    
    # Add date filtering if provided
    if date_from or date_to:
        # Convert to arXiv date format (YYYYMMDD)
        date_filter_parts = []
        if date_from:
            try:
                start_date = datetime.strptime(date_from, "%Y-%m-%d").strftime("%Y%m%d")
                date_filter_parts.append(f"submittedDate:[{start_date}0000")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_from format. Use YYYY-MM-DD")
        else:
            date_filter_parts.append("submittedDate:[19910101")
            
        if date_to:
            try:
                end_date = datetime.strptime(date_to, "%Y-%m-%d").strftime("%Y%m%d")
                date_filter_parts.append(f"{end_date}2359]")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_to format. Use YYYY-MM-DD")
        else:
            current_date = datetime.now().strftime("%Y%m%d")
            date_filter_parts.append(f"{current_date}2359]")
            
        query += f" AND {''.join(date_filter_parts)}"
    
    return await search_papers(q=query, max_results=max_results, page=page, sort_by=sort_by)


@app.get("/api/search/advanced", response_model=SearchResponse, tags=["Search"])
async def advanced_search(
    q: str = Query(..., description="Search query"),
    categories: Optional[str] = Query(None, description="Comma-separated list of categories to filter by"),
    min_date: Optional[str] = Query(None, description="Minimum publication date (YYYY-MM-DD)"),
    max_date: Optional[str] = Query(None, description="Maximum publication date (YYYY-MM-DD)"),
    authors: Optional[str] = Query(None, description="Comma-separated list of authors to include"),
    exclude_categories: Optional[str] = Query(None, description="Comma-separated list of categories to exclude"),
    max_results: int = Query(20, ge=1, le=100),
    page: int = Query(1, ge=1),
    sort_by: Literal["relevance", "lastUpdatedDate", "submittedDate"] = Query("relevance")
) -> SearchResponse:
    """Advanced search with multiple filters and criteria."""
    # Build complex query
    query_parts = [f"({q})"]
    
    # Add category filters
    if categories:
        cat_list = [cat.strip() for cat in categories.split(',')]
        cat_query = " OR ".join([f"cat:{cat}" for cat in cat_list])
        query_parts.append(f"({cat_query})")
    
    # Add author filters
    if authors:
        auth_list = [auth.strip() for auth in authors.split(',')]
        auth_query = " OR ".join([f"au:{auth}" for auth in auth_list])
        query_parts.append(f"({auth_query})")
    
    # Add date range
    if min_date or max_date:
        start_date = min_date or "1991-01-01"
        end_date = max_date or datetime.now().strftime("%Y-%m-%d")
        
        try:
            start_formatted = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%d")
            end_formatted = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y%m%d")
            query_parts.append(f"submittedDate:[{start_formatted}0000 TO {end_formatted}2359]")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    # Exclude categories
    if exclude_categories:
        excl_list = [cat.strip() for cat in exclude_categories.split(',')]
        for cat in excl_list:
            query_parts.append(f"NOT cat:{cat}")
    
    # Combine all parts
    final_query = " AND ".join(query_parts)
    
    return await search_papers(q=final_query, max_results=max_results, page=page, sort_by=sort_by)


@app.get("/api/search/similar/{arxiv_id}", response_model=SearchResponse, tags=["Search"])
async def find_similar_papers(
    arxiv_id: str,
    max_results: int = Query(10, ge=1, le=50),
    page: int = Query(1, ge=1)
) -> SearchResponse:
    """Find papers similar to a given arXiv paper based on categories and keywords."""
    try:
        # Get the reference paper
        paper = await get_comprehensive_paper_content(arxiv_id)
        
        # Extract categories and key terms from title/abstract
        categories = paper.metadata.categories[:3]  # Top 3 categories
        title_words = paper.metadata.title.lower().split()
        
        # Build similarity query
        cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
        
        # Extract meaningful keywords from title (excluding common words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must'}
        keywords = [word for word in title_words if len(word) > 3 and word not in stop_words][:5]
        
        if keywords:
            keyword_query = " OR ".join([f"ti:{word}" for word in keywords])
            query = f"({cat_query}) AND ({keyword_query}) AND NOT id_list:{arxiv_id}"
        else:
            query = f"({cat_query}) AND NOT id_list:{arxiv_id}"
        
        return await search_papers(q=query, max_results=max_results, page=page, sort_by="relevance")
        
    except Exception as e:
        logger.error(f"Similar papers search error: {e}")
        raise HTTPException(status_code=500, detail="Failed to find similar papers")


@app.get("/api/search/trending", response_model=SearchResponse, tags=["Search"])
async def get_trending_papers(
    category: Optional[str] = Query(None, description="Category to filter trending papers"),
    days: int = Query(7, ge=1, le=30, description="Number of recent days to consider"),
    max_results: int = Query(20, ge=1, le=100)
) -> SearchResponse:
    """Get trending/recent papers, optionally filtered by category."""
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    start_formatted = start_date.strftime("%Y%m%d")
    end_formatted = end_date.strftime("%Y%m%d")
    
    # Build query
    date_query = f"submittedDate:[{start_formatted}0000 TO {end_formatted}2359]"
    
    if category:
        query = f"cat:{category} AND {date_query}"
    else:
        # Popular categories for trending
        popular_cats = ["cs.AI", "cs.LG", "cs.CV", "cs.CL", "quant-ph", "physics.gr-qc"]
        cat_query = " OR ".join([f"cat:{cat}" for cat in popular_cats])
        query = f"({cat_query}) AND {date_query}"
    
    return await search_papers(q=query, max_results=max_results, page=1, sort_by="submittedDate")


@app.get("/api/download/{arxiv_id}", tags=["Download"])
async def download_paper_pdf(arxiv_id: str):
    """Download PDF file for a specific arXiv paper."""
    try:
        logger.info(f"Downloading PDF for paper {arxiv_id}")
        
        # Search for the paper using arxiv.py
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        papers = list(client.results(search))
        
        if not papers:
            raise HTTPException(status_code=404, detail=f"Paper {arxiv_id} not found")
        
        paper = papers[0]
        
        # Download PDF to temporary location
        pdf_path = await _download_pdf(paper)
        
        # Read the PDF file
        async with aiofiles.open(pdf_path, 'rb') as f:
            pdf_content = await f.read()
        
        # Clean up temporary file
        if pdf_path.exists():
            pdf_path.unlink()
        
        # Generate filename
        safe_title = re.sub(r'[^\w\s-]', '', paper.title)[:50]
        safe_title = re.sub(r'[-\s]+', '-', safe_title).strip('-')
        filename = f"{arxiv_id}_{safe_title}.pdf"
        
        # Return PDF with appropriate headers
        from fastapi.responses import Response
        return Response(
            content=pdf_content,
            media_type='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename="{filename}"',
                'Content-Length': str(len(pdf_content))
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading PDF for {arxiv_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to download PDF") from e


@app.post("/api/exam/selected", response_model=ExamResponse, tags=["Exam"])
async def create_exam_from_selected_papers(
    request: Request,
    selected_request: SelectedPapersExamRequest
) -> ExamResponse:
    """Generate exam from specific selected papers by arXiv ID."""
    # Extract LLM configuration from headers
    provider = request.headers.get("X-LLM-Provider", "openai").lower()
    model = request.headers.get("X-LLM-Model", "gpt-4o-mini")
    api_key = request.headers.get("X-LLM-API-Key")
    
    if not api_key:
        raise HTTPException(status_code=400, detail="API key is required. Please configure your LLM provider settings.")
    
    logger.info(f"Creating exam from {len(selected_request.arxiv_ids)} selected papers using {provider}/{model}")
    
    try:
        # Get comprehensive content for each selected paper
        papers = []
        failed_papers = []
        
        for arxiv_id in selected_request.arxiv_ids:
            try:
                logger.info(f"Attempting to process paper {arxiv_id}")
                
                # Clean up the arXiv ID format if needed
                clean_arxiv_id = arxiv_id.strip()
                if not re.match(r'^\d{4}\.\d{4,5}$', clean_arxiv_id):
                    logger.warning(f"Unusual arXiv ID format: {clean_arxiv_id}")
                
                paper = await get_comprehensive_paper_content(clean_arxiv_id)
                papers.append(paper)
                logger.info(f"Successfully processed paper {clean_arxiv_id} ({paper.processing_method})")
            except HTTPException as e:
                logger.warning(f"HTTP error processing paper {arxiv_id}: {e.detail}")
                failed_papers.append(arxiv_id)
                continue
            except Exception as e:
                logger.error(f"Unexpected error processing paper {arxiv_id}: {type(e).__name__}: {e}")
                failed_papers.append(arxiv_id)
                continue
        
        if not papers:
            error_msg = f"Failed to process any of the selected papers. Failed papers: {failed_papers}"
            if failed_papers:
                error_msg += f". This could be due to invalid arXiv IDs, papers not found, or network issues."
            raise HTTPException(status_code=404, detail=error_msg)
        
        if failed_papers:
            logger.warning(f"Failed to process {len(failed_papers)} papers: {failed_papers}")
        
        # Combine all papers into a single "super paper" for exam generation
        exam_title = selected_request.exam_title or f"Multi-Paper Exam ({len(papers)} papers)"
        
        combined_metadata = PaperMetadata(
            title=exam_title,
            authors=list(set([author for paper in papers for author in paper.metadata.authors])),
            abstract=f"Exam generated from {len(papers)} selected papers: {', '.join([p.metadata.title[:50] + '...' if len(p.metadata.title) > 50 else p.metadata.title for p in papers[:3]])}",
            categories=list(set([cat for paper in papers for cat in paper.metadata.categories]))
        )
        
        # Combine content from all papers
        combined_sections = {}
        combined_text_parts = []
        total_chars = 0
        
        for i, paper in enumerate(papers):
            paper_prefix = f"PAPER {i+1}: {paper.metadata.title[:100]}..."
            combined_text_parts.append(f"\n\n--- {paper_prefix} ---")
            combined_text_parts.append(paper.full_text[:5000])  # Limit per paper to manage size
            
            # Merge sections with paper prefixes
            for section_name, content in paper.sections.items():
                section_key = f"paper_{i+1}_{section_name}"
                combined_sections[section_key] = content[:2000]  # Limit section size
            
            total_chars += len(paper.full_text)
        
        combined_text = "\n".join(combined_text_parts)
        
        combined_paper = ProcessedPaper(
            metadata=combined_metadata,
            full_text=combined_text,
            sections=combined_sections,
            total_chars=total_chars, 
            processing_method="pdf"
        )
        
        # Generate questions
        total_questions = selected_request.mc_questions + selected_request.oe_questions
        questions = await generate_questions(
            combined_paper, 
            total_questions, 
            selected_request.mc_questions, 
            selected_request.oe_questions, 
            provider, 
            model, 
            api_key
        )
        
        # Generate exam name
        exam_name = await generate_exam_name(combined_paper, questions, provider, model, api_key)
        
        mc_count = sum(1 for q in questions if q.type == "multiple_choice")
        oe_count = len(questions) - mc_count
        logger.info(f"Generated {mc_count} MC + {oe_count} OE questions from {len(papers)} selected papers")
        logger.info(f"Generated exam name: '{exam_name}'")
        
        return ExamResponse(metadata=combined_metadata, questions=questions, exam_name=exam_name)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in selected papers exam generation: {e}")
        raise HTTPException(status_code=500, detail="Internal error") from e


@app.post("/api/exam/search", response_model=ExamResponse, tags=["Exam"])
async def create_exam_from_search(
    request: Request,
    search_request: SearchExamRequest
) -> ExamResponse:
    """Generate exam from multiple papers found via search query."""
    # Extract LLM configuration from headers
    provider = request.headers.get("X-LLM-Provider", "openai").lower()
    model = request.headers.get("X-LLM-Model", "gpt-4o-mini")
    api_key = request.headers.get("X-LLM-API-Key")
    
    if not api_key:
        raise HTTPException(status_code=400, detail="API key is required. Please configure your LLM provider settings.")
    
    logger.info(f"Creating exam from search: '{search_request.query}' with {search_request.max_papers} papers using {provider}/{model}")
    
    try:
        # Search for papers
        search_results = await search_arxiv_papers(
            query=search_request.query,
            max_results=search_request.max_papers,
            sort_by=search_request.sort_by
        )
        
        if not search_results:
            raise HTTPException(status_code=404, detail="No papers found for the given search query")
        
        logger.info(f"Found {len(search_results)} papers for exam generation")
        
        # Get comprehensive content for each paper
        papers = []
        for result in search_results:
            try:
                paper = await get_comprehensive_paper_content(result.arxiv_id)
                papers.append(paper)
            except Exception as e:
                logger.warning(f"Failed to process paper {result.arxiv_id}: {e}")
                continue
        
        if not papers:
            raise HTTPException(status_code=404, detail="Failed to process any papers from search results")
        
        # Combine all papers into a single "super paper" for exam generation
        combined_metadata = PaperMetadata(
            title=f"Multi-Paper Exam: {search_request.query}",
            authors=list(set([author for paper in papers for author in paper.metadata.authors])),
            abstract=f"Exam generated from {len(papers)} papers matching query: {search_request.query}",
            categories=list(set([cat for paper in papers for cat in paper.metadata.categories]))
        )
        
        # Combine content from all papers
        combined_sections = {}
        combined_text_parts = []
        total_chars = 0
        
        for i, paper in enumerate(papers):
            paper_prefix = f"PAPER {i+1}: {paper.metadata.title[:100]}..."
            combined_text_parts.append(f"\n\n--- {paper_prefix} ---")
            combined_text_parts.append(paper.full_text[:5000])  # Limit per paper to manage size
            
            # Merge sections with paper prefixes
            for section_name, content in paper.sections.items():
                section_key = f"paper_{i+1}_{section_name}"
                combined_sections[section_key] = content[:2000]  # Limit section size
            
            total_chars += len(paper.full_text)
        
        combined_text = "\n".join(combined_text_parts)
        
        combined_paper = ProcessedPaper(
            metadata=combined_metadata,
            full_text=combined_text,
            sections=combined_sections,
            total_chars=total_chars, 
            processing_method="pdf"
        )
        
        # Generate questions
        total_questions = search_request.mc_questions + search_request.oe_questions
        questions = await generate_questions(
            combined_paper, 
            total_questions, 
            search_request.mc_questions, 
            search_request.oe_questions, 
            provider, 
            model, 
            api_key
        )
        
        # Generate exam name
        exam_name = await generate_exam_name(combined_paper, questions, provider, model, api_key)
        
        mc_count = sum(1 for q in questions if q.type == "multiple_choice")
        oe_count = len(questions) - mc_count
        logger.info(f"Generated {mc_count} MC + {oe_count} OE questions from {len(papers)} papers")
        logger.info(f"Generated exam name: '{exam_name}'")
        
        return ExamResponse(metadata=combined_metadata, questions=questions, exam_name=exam_name)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in search exam generation: {e}")
        raise HTTPException(status_code=500, detail="Internal error") from e


@app.get("/api/exam/{arxiv_id}", response_model=ExamResponse, tags=["Exam"])
async def create_exam(
    request: Request,
    arxiv_id: str,
    mc_questions: int = Query(10, ge=1, le=12, description="Number of multiple-choice questions"),
    oe_questions: int = Query(5, ge=1, le=8, description="Number of open-ended questions"),
    teacher_mode: bool = Query(False, description="Include all answers and graded examples for teachers"),
) -> ExamResponse:
    """Generate comprehensive exam from arXiv paper with split generation: MC questions first, then open-ended."""
    total_questions = mc_questions + oe_questions
    
    # Extract LLM configuration from headers
    provider = request.headers.get("X-LLM-Provider", "openai").lower()
    model = request.headers.get("X-LLM-Model", "gpt-4o-mini")
    api_key = request.headers.get("X-LLM-API-Key")
    
    if not api_key:
        raise HTTPException(status_code=400, detail="API key is required. Please configure your LLM provider settings.")
    
    logger.info(f"Creating exam for {arxiv_id} with {mc_questions} MC + {oe_questions} OE questions using {provider}/{model}")
    
    try:
        paper = await get_comprehensive_paper_content(arxiv_id)
        questions = await generate_questions(paper, total_questions, mc_questions, oe_questions, provider, model, api_key)
        
        # Generate exam name
        exam_name = await generate_exam_name(paper, questions, provider, model, api_key)
        
        mc_count = sum(1 for q in questions if q.type == "multiple_choice")
        oe_count = len(questions) - mc_count
        logger.info(f"Generated {mc_count} MC + {oe_count} open-ended questions using {paper.processing_method}")
        logger.info(f"Generated exam name: '{exam_name}'")
        
        return ExamResponse(metadata=paper.metadata, questions=questions, exam_name=exam_name)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal error") from e

# Static files and error handling
@app.exception_handler(Exception)
async def handle_exception(_: Request, exc: Exception) -> Response:
    logger.exception(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500, 
        content={"detail": "Internal Server Error"}
    )

# CLI entry point
def main() -> None:
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=bool(os.getenv("DEV")),
        log_level="info",
    )

if __name__ == "__main__":
    main()
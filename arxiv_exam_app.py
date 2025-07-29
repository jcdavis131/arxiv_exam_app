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
import re
import sys
import tempfile
from functools import lru_cache
from typing import List, Literal, Optional, Dict, Any, Union
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from fastapi.staticfiles import StaticFiles
import aiofiles

# Configuration Constants
ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_PDF_URL = "http://arxiv.org/pdf"
ARXIV_TIMEOUT = 3.0
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
    choices: List[Choice] = Field(..., min_items=2)
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


class ProcessedPaper(BaseModel):
    metadata: PaperMetadata
    full_text: str
    sections: Dict[str, str] = {}  # section_name -> content
    total_chars: int
    processing_method: Literal["pdf", "abstract_only"] = "pdf"


class ExamResponse(BaseModel):
    metadata: PaperMetadata
    questions: List[Question]

# FastAPI Application
app = FastAPI(
    title="arXiv Exam Generator",
    description="Generate comprehensive multiple-choice exams from arXiv papers using RAG",
    version="1.0.0",
    license_info={"name": "MIT"},
    contact={"url": "https://github.com/jcdavis131/arxiv-exam-app"},
)

# ---------------------------------------------------------------------------
# Helper funcs – arXiv Processing
# ---------------------------------------------------------------------------
@lru_cache(maxsize=256)
def _parse_arxiv_metadata(atom: str) -> PaperMetadata:
    """Parse arXiv atom feed to extract paper metadata."""
    try:
        import feedparser
    except ImportError:
        sys.exit("pip install feedparser >=6.0")

    feed = feedparser.parse(atom)
    if not feed.entries:
        raise ValueError("No entries in arXiv feed")
    
    entry = feed.entries[0]
    authors = [author.name for author in getattr(entry, 'authors', [])]
    categories = [tag.term for tag in getattr(entry, 'tags', [])]
    
    return PaperMetadata(
        title=_normalize_whitespace(entry.title),
        authors=authors,
        abstract=_normalize_whitespace(entry.summary),
        published=getattr(entry, 'published', None),
        categories=categories
    )


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    return re.sub(r"\s+", " ", text).strip()


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


async def _download_pdf(arxiv_id: str) -> Path:
    """Download PDF from arXiv."""
    pdf_url = f"{ARXIV_PDF_URL}/{arxiv_id}"
    
    async with httpx.AsyncClient(timeout=ARXIV_TIMEOUT) as client:
        try:
            response = await client.get(pdf_url)
            response.raise_for_status()
            
            if len(response.content) > MAX_PDF_SIZE:
                raise ValueError(f"PDF too large: {len(response.content)} bytes")
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
            temp_path = Path(temp_file.name)
            
            async with aiofiles.open(temp_path, 'wb') as f:
                await f.write(response.content)
            
            return temp_path
            
        except httpx.HTTPError as exc:
            logger.error(f"PDF download failed for {arxiv_id}: {exc}")
            raise HTTPException(status_code=502, detail="Failed to download PDF") from exc


async def get_comprehensive_paper_content(arxiv_id: str) -> ProcessedPaper:
    """Get comprehensive paper content including full PDF text."""
    # First get basic metadata
    async with httpx.AsyncClient(timeout=ARXIV_TIMEOUT) as client:
        try:
            r = await client.get(ARXIV_API_URL, params={"id_list": arxiv_id})
            r.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("arXiv API error: %s", exc)
            raise HTTPException(status_code=502, detail="arXiv API error") from exc
    
    try:
        metadata = _parse_arxiv_metadata(r.text)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    
    # Try to get full PDF content
    pdf_path = None
    try:
        logger.info(f"Downloading PDF for {arxiv_id}")
        pdf_path = await _download_pdf(arxiv_id)
        
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
                questions.append(question)
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


# ---------------------------------------------------------------------------
# API route
# ---------------------------------------------------------------------------
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
        
        mc_count = sum(1 for q in questions if q.type == "multiple_choice")
        oe_count = len(questions) - mc_count
        logger.info(f"Generated {mc_count} MC + {oe_count} open-ended questions using {paper.processing_method}")
        
        return ExamResponse(metadata=paper.metadata, questions=questions)
        
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
        "arxiv_exam_app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=bool(os.getenv("DEV")),
        log_level="info",
    )

if __name__ == "__main__":
    main()

# Mount static files only if not running on Vercel
if not os.getenv("VERCEL"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
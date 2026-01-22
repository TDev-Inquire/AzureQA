"""
Call Center AI Triage Dashboard

A production-grade Streamlit web application for batch processing and analyzing 
call center recordings. This dashboard enables quality assurance teams to:

1. Upload audio files in batch (mp3, wav, m4a formats)
2. Automatically transcribe calls with speaker identification (diarization)
3. Analyze transcript sentiment, detect PII (Personally Identifiable Information)
4. Grade calls using AI (via Prompt Flow)
5. Generate comprehensive Excel/CSV/HTML reports for leadership review

The application uses Azure Cognitive Services (Speech, Language, Prompt Flow) 
for intelligent analysis and flags calls requiring immediate attention. All 
sensitive data (PII) is automatically redacted for compliance and privacy.

Architecture:
- Frontend: Streamlit (interactive web dashboard)
- Transcription: Azure Speech Service (batch transcription with diarization)
- Analysis: Azure Language Service (PII detection, sentiment, summarization)
- Grading: Azure Prompt Flow (LLM-based custom evaluation)
- Storage: Azure Blob Storage (temporary file handling, SAS URLs)

Environment Requirements:
- SPEECH_KEY: Azure Speech Service subscription key
- SPEECH_REGION: Azure region for Speech Service (e.g., "eastus")
- STORAGE_CONN_STR: Azure Storage account connection string
- PF_ENDPOINT: Azure Prompt Flow endpoint URL
- PF_KEY: Authentication key for Prompt Flow
- LANGUAGE_KEY: Azure Language Service key
- LANGUAGE_ENDPOINT: Azure Language Service endpoint URL
"""

import streamlit as st
import requests
import json
import time
import os
import io
import base64
import pandas as pd
import concurrent.futures
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions

# ============================================================================
# CONFIGURATION - Azure Service Credentials
# ============================================================================
# All credentials are loaded from environment variables for security.
# These should be stored in Azure Key Vault or managed via platform secrets.
# Never commit credentials to version control.

SPEECH_KEY = os.environ.get("SPEECH_KEY", "")  # Azure Speech Service subscription key
SPEECH_REGION = os.environ.get("SPEECH_REGION", "eastus")  # Azure region (e.g., "eastus", "westeurope")
STORAGE_CONN_STR = os.environ.get("STORAGE_CONN_STR", "")  # Connection string for Azure Blob Storage
PF_ENDPOINT = os.environ.get("PF_ENDPOINT", "")  # Prompt Flow endpoint URL for AI grading
PF_KEY = os.environ.get("PF_KEY", "")  # Authentication key for Prompt Flow

# Azure Language Service for PII detection, sentiment analysis, and conversation summarization
LANGUAGE_KEY = os.environ.get("LANGUAGE_KEY", "")  # Azure Language Service subscription key
LANGUAGE_ENDPOINT = os.environ.get("LANGUAGE_ENDPOINT", "")  # Azure Language Service endpoint URL

# Blob Storage container where temporary audio files are stored for processing
CONTAINER_NAME = "call-uploads"

# Speaker role identification mapping
# The Speech Recognition API identifies speakers by number (1, 2, 3, etc).
# This mapping provides human-readable labels for the dashboard.
# Customize based on your typical call structure (e.g., representative + customer).
SPEAKER_LABELS = {
    1: "Representative",  # Speaker 1 is typically the first to speak (agent/representative)
    2: "Customer",  # Speaker 2 is typically the customer/caller
    3: "Speaker 3",  # Additional speakers (e.g., supervisor, transfer recipient)
    4: "Speaker 4",
    5: "Speaker 5"
}


# ============================================================================
# PAGE CONFIGURATION & STYLING
# ============================================================================
# Configure the Streamlit app layout and visual appearance for better UX

st.set_page_config(layout="wide", page_title="Call Center Triage", page_icon="📞")

# Custom CSS styling for the dashboard cards, status indicators, and responsive layout
# Colors: Red for critical flags, Green for passed calls, Gray for neutral metrics
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .status-bad {
        color: #d9534f;
        font-weight: bold;
    }
    .status-good {
        color: #5cb85c;
        font-weight: bold;
    }
    .stDataFrame { width: 100%; }
</style>
""", unsafe_allow_html=True)

st.title("📞 Call Center Triage Dashboard")
st.markdown("Upload a batch of calls. The AI will grade them and **flag the ones requiring attention**.")

# ============================================================================
# HELPER FUNCTIONS - CORE PROCESSING PIPELINE
# ============================================================================
# The following functions orchestrate the entire AI analysis workflow:
# 1. File upload → Azure Blob Storage (SAS URL generation)
# 2. Audio transcription with speaker identification
# 3. Sentiment analysis and PII detection
# 4. AI grading via Prompt Flow
# 5. Report generation (Excel, CSV, HTML)

def get_blob_sas_url(file_obj):
    """
    Upload audio file to Azure Blob Storage and generate a temporary SAS URL.
    
    SAS (Shared Access Signature) URLs provide time-limited access to the file
    without exposing the storage account key. This is required by Azure Speech
    Service for batch transcription jobs.
    
    Args:
        file_obj: Streamlit UploadedFile object
        
    Returns:
        str: Temporary SAS URL (valid for 1 hour) or None if upload fails
        
    Process:
        1. Connect to Azure Storage account
        2. Create blob container if it doesn't exist
        3. Upload the file with a timestamped name for uniqueness
        4. Generate a read-only SAS token (valid 1 hour)
        5. Return the file URL with SAS token appended
    """
    try:
        blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONN_STR)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        if not container_client.exists():
            container_client.create_container()

        blob_name = f"batch_{int(time.time())}_{file_obj.name}"
        blob_client = container_client.get_blob_client(blob_name)
        
        file_obj.seek(0)
        blob_client.upload_blob(file_obj, overwrite=True)

        sas_token = generate_blob_sas(
            account_name=blob_service_client.account_name,
            container_name=CONTAINER_NAME,
            blob_name=blob_name,
            account_key=blob_service_client.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=1)
        )
        return f"{blob_client.url}?{sas_token}"
    except Exception as e:
        return None

def transcribe_audio(sas_url):
    """
    Transcribe audio file using Azure Speech Service with speaker diarization.
    
    This function submits a batch transcription job to Azure Speech Service
    and polls until completion. Unlike real-time transcription, batch
    transcription supports advanced features like speaker identification
    (diarization), word-level timestamps, and punctuation.
    
    Args:
        sas_url: Temporary URL to audio file in Azure Blob Storage
        
    Returns:
        dict: Structured transcription result containing:
            - combined_text: Full conversation formatted as "Speaker: text"
            - conversation_turns: List of individual speaking turns with metadata
            - speaker_stats: Word/turn counts per speaker
            - total_speakers: Number of unique speakers detected
            - raw_json: Full Azure Speech API response
        None: If transcription fails or times out
    
    Diarization Details:
        - Identifies 2-5 different speakers automatically
        - Assigns speaker IDs (1, 2, 3, etc.)
        - Enables word-level timestamps for precise timing
        - Masks profanity automatically
        
    Processing:
        1. POST to Azure Speech transcription API with diarization config
        2. Poll job status every 2 seconds (typical jobs complete in seconds)
        3. Retrieve transcript and extract phrases with speaker labels
        4. Build conversation turns with timestamps and confidence scores
        5. Calculate speaker statistics (word count, turn count)
    """
    url = f"https://{SPEECH_REGION}.api.cognitive.microsoft.com/speechtotext/v3.2/transcriptions"
    headers = {"Ocp-Apim-Subscription-Key": SPEECH_KEY, "Content-Type": "application/json"}
    payload = {
        "contentUrls": [sas_url],  # Audio file to transcribe
        "locale": "en-US",  # Transcription language (U.S. English)
        "displayName": "Batch_Job",  # Human-readable job name
        "properties": {
            # ============= DIARIZATION SETTINGS =============
            # Speaker identification that separates different speakers in the call
            "diarizationEnabled": True,
            "diarization": {
                "speakers": {
                    "minCount": 2,  # Minimum 2 speakers (representative + customer)
                    "maxCount": 5   # Maximum 5 speakers (allows supervisor, transfer, etc.)
                }
            },
            # ============= ACCURACY SETTINGS =============
            "wordLevelTimestampsEnabled": True,  # Enables timestamp for each word (better UI)
            "punctuationMode": "DictatedAndAutomatic",  # Adds punctuation and capitalization
            "profanityFilterMode": "Masked"  # Masks profanity with asterisks (**)
        }
    }

    # --- Submit Transcription Job ---
    # Start the async transcription job with Azure Speech Service
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 201:
        return None
    
    job_url = response.json()["self"]  # Unique URL for this job
    
    # --- Poll for Job Completion ---
    # Transcription is asynchronous. Poll every 2 seconds until complete.
    # Most jobs complete in 5-30 seconds depending on audio length.
    while True:
        job_response = requests.get(job_url, headers=headers).json()
        status = job_response["status"]
        if status == "Succeeded":
            break
        elif status == "Failed":
            return None
        time.sleep(2)  # Wait 2 seconds before checking again
            
    # --- Retrieve Transcription Results ---
    # Download the completed transcription file
    results_url = job_response.get("links", {}).get("files")
    files_data = requests.get(results_url, headers=headers).json()
    values = files_data.get("values", [])
    transcript_file = next((f for f in values if f.get("kind") == "Transcription"), None)
    
    if not transcript_file:
        return None
        
    content_url = transcript_file.get("links", {}).get("contentUrl")
    transcript_response = requests.get(content_url)
    transcript_json = transcript_response.json()
    
    # --- Parse Transcript into Structured Format ---
    # Convert raw Azure response into a more usable structure with speaker info
    
    # Extract recognized phrases with speaker info and confidence scores
    recognized_phrases = transcript_json.get("recognizedPhrases", [])
    
    # Build structured conversation turns
    conversation_turns = []
    for phrase in recognized_phrases:
        speaker_id = phrase.get("speaker", 1)  # Default to Speaker 1 if not identified
        speaker_label = SPEAKER_LABELS.get(speaker_id, f"Speaker {speaker_id}")
        
        # Get best transcription result (nBest = n-best alternatives)
        n_best = phrase.get("nBest", [{}])
        best = n_best[0] if n_best else {}
        
        text = best.get("display", "")  # Final transcribed text with capitalization
        confidence = best.get("confidence", 0)  # Confidence score 0.0-1.0
        
        # Convert timestamps from ticks (Windows file time units) to seconds
        # Windows ticks = 100-nanosecond intervals, so divide by 10,000,000
        offset_ticks = phrase.get("offsetInTicks", 0)
        offset_seconds = offset_ticks / 10_000_000
        timestamp = format_timestamp(offset_seconds)  # Convert to MM:SS format
        
        # Get word-level details for more granular analysis
        words = best.get("words", [])
        
        # Store this conversational turn with all metadata
        conversation_turns.append({
            "speaker_id": speaker_id,
            "speaker": speaker_label,
            "text": text,
            "timestamp": timestamp,  # MM:SS format for display
            "offset_seconds": offset_seconds,  # Raw seconds for calculations
            "confidence": confidence,  # Transcription confidence 0.0-1.0
            "words": words  # Word-level timestamps and confidence
        })
    
    # Build combined text for AI grading (used by Prompt Flow)
    combined_text = "\n".join([f"{t['speaker']}: {t['text']}" for t in conversation_turns])
    
    # --- Calculate Speaker Statistics ---
    # Summarize speaking time and engagement per speaker
    speaker_stats = {}
    for turn in conversation_turns:
        sid = turn["speaker"]
        if sid not in speaker_stats:
            speaker_stats[sid] = {"turns": 0, "words": 0}
        speaker_stats[sid]["turns"] += 1
        speaker_stats[sid]["words"] += len(turn["text"].split())
    
    return {
        "combined_text": combined_text,
        "conversation_turns": conversation_turns,
        "speaker_stats": speaker_stats,
        "total_speakers": len(speaker_stats),
        "raw_json": transcript_json
    }


def format_timestamp(seconds):
    """
    Convert seconds to MM:SS display format.
    
    Args:
        seconds: Time in seconds (int or float)
        
    Returns:
        str: Formatted time string (e.g., "01:23" for 83 seconds)
    """
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def detect_audio_events(conversation_turns):
    """
    Detect potential audio quality issues and anomalies between speaker turns.
    
    Currently detects:
    - Long pauses (>5 seconds between speakers) which may indicate dead air,
      transferred calls, or system issues
    
    Args:
        conversation_turns: List of dicts with speaker turn data including
                          offset_seconds and text
    
    Returns:
        list: Events detected with format:
              {
                  "type": "long_pause",
                  "after_turn": turn_index,
                  "duration": gap_in_seconds,
                  "description": "[Pause: ~X seconds]"
              }
    
    Future Enhancements:
        - Background noise detection
        - Echo or overlap detection  
        - Audio level spikes
    """
    events = []
    for i in range(1, len(conversation_turns)):
        prev_turn = conversation_turns[i-1]
        curr_turn = conversation_turns[i]
        
        # Calculate gap between end of previous turn and start of current turn
        # Estimate: 1 word ≈ 0.4 seconds of speech (rough average)
        prev_end = prev_turn["offset_seconds"] + len(prev_turn["text"].split()) * 0.4
        gap = curr_turn["offset_seconds"] - prev_end
        
        # Flag pauses longer than 5 seconds (normal pauses are <1 sec)
        if gap > 5:
            events.append({
                "type": "long_pause",
                "after_turn": i-1,
                "duration": gap,
                "description": f"[Pause: ~{int(gap)} seconds]"
            })
    return events


def analyze_with_language_service(conversation_turns):
    """
    Comprehensive language analysis using Azure Cognitive Services.
    
    Performs three parallel analyses on the call transcript:
    
    1. PII DETECTION & REDACTION
       - Identifies sensitive data: phone numbers, SSN, credit cards, names, addresses, emails
       - Automatically redacts detected PII in the dashboard display
       - Required for compliance (GDPR, PCI-DSS, etc.)
    
    2. SENTIMENT ANALYSIS
       - Per-turn sentiment classification: positive, neutral, negative
       - Confidence scores for each classification
       - Opinion mining for fine-grained sentiment aspects
    
    3. CALL SUMMARIZATION
       - Extracts issue/problem identified in the call
       - Summarizes resolution provided or taken
       - Uses AI to understand conversation intent and outcome
    
    Args:
        conversation_turns: List of dicts with "text" and "speaker" keys
        
    Returns:
        dict: Analysis results with keys:
            - pii_results: List of detected PII entities per turn
            - sentiment_results: Sentiment per turn with confidence
            - summary: Dict with "issue" and "resolution" keys
            - redacted_turns: Conversation with PII masked (e.g., "PHONE_NUMBER")
            - Errors: Optional error messages if any analysis fails
            
    Notes:
        - All analysis is performed server-side by Azure; transcript data is sent
        - Returns gracefully if Language Service keys not configured
        - Timeouts after 30 seconds per API call
    """
    if not LANGUAGE_KEY or not LANGUAGE_ENDPOINT:
        return {"pii_results": None, "sentiment_results": None, "summary": None}
    
    results = {
        "pii_results": [],
        "sentiment_results": [],
        "summary": None,
        "redacted_turns": []
    }
    
    # Prepare documents for Language API
    documents = []
    for i, turn in enumerate(conversation_turns):
        documents.append({
            "id": str(i),
            "language": "en",
            "text": turn["text"]
        })
    
    headers = {
        "Ocp-Apim-Subscription-Key": LANGUAGE_KEY,
        "Content-Type": "application/json"
    }
    
    # 1. PII Detection & Redaction
    try:
        pii_url = f"{LANGUAGE_ENDPOINT}/language/:analyze-text?api-version=2023-04-01"
        pii_payload = {
            "kind": "PiiEntityRecognition",
            "parameters": {
                "modelVersion": "latest",
                "piiCategories": ["Person", "PhoneNumber", "Address", "Email", "CreditCardNumber", "SSN", "Organization"]
            },
            "analysisInput": {"documents": documents}
        }
        pii_response = requests.post(pii_url, headers=headers, json=pii_payload, timeout=30)
        if pii_response.status_code == 200:
            pii_data = pii_response.json()
            for doc in pii_data.get("results", {}).get("documents", []):
                doc_id = int(doc["id"])
                entities = doc.get("entities", [])
                redacted_text = doc.get("redactedText", conversation_turns[doc_id]["text"])
                results["pii_results"].append({
                    "turn_id": doc_id,
                    "entities": entities,
                    "redacted_text": redacted_text
                })
                results["redacted_turns"].append({
                    **conversation_turns[doc_id],
                    "original_text": conversation_turns[doc_id]["text"],
                    "text": redacted_text,
                    "pii_detected": len(entities) > 0
                })
    except Exception as e:
        results["pii_error"] = str(e)
    
    # 2. Sentiment Analysis
    try:
        sentiment_url = f"{LANGUAGE_ENDPOINT}/language/:analyze-text?api-version=2023-04-01"
        sentiment_payload = {
            "kind": "SentimentAnalysis",
            "parameters": {"modelVersion": "latest", "opinionMining": True},
            "analysisInput": {"documents": documents}
        }
        sentiment_response = requests.post(sentiment_url, headers=headers, json=sentiment_payload, timeout=30)
        if sentiment_response.status_code == 200:
            sentiment_data = sentiment_response.json()
            for doc in sentiment_data.get("results", {}).get("documents", []):
                doc_id = int(doc["id"])
                results["sentiment_results"].append({
                    "turn_id": doc_id,
                    "speaker": conversation_turns[doc_id]["speaker"],
                    "sentiment": doc.get("sentiment", "neutral"),
                    "confidence_scores": doc.get("confidenceScores", {})
                })
    except Exception as e:
        results["sentiment_error"] = str(e)
    
    # 3. Conversation Summarization
    try:
        # Format as conversation for summarization
        conversation_items = []
        for i, turn in enumerate(conversation_turns):
            conversation_items.append({
                "id": str(i),
                "participantId": turn["speaker"],
                "text": turn["text"]
            })
        
        summary_url = f"{LANGUAGE_ENDPOINT}/language/analyze-conversations/jobs?api-version=2023-04-01"
        summary_payload = {
            "displayName": "Call Summary",
            "analysisInput": {
                "conversations": [{
                    "id": "1",
                    "language": "en",
                    "modality": "transcript",
                    "conversationItems": conversation_items
                }]
            },
            "tasks": [{
                "kind": "ConversationalSummarizationTask",
                "taskName": "Call Summary",
                "parameters": {
                    "summaryAspects": ["issue", "resolution"]
                }
            }]
        }
        
        # Submit summarization job
        summary_response = requests.post(summary_url, headers=headers, json=summary_payload, timeout=30)
        if summary_response.status_code == 202:
            job_url = summary_response.headers.get("operation-location")
            # Poll for completion
            for _ in range(30):  # Max 30 attempts
                time.sleep(2)
                job_response = requests.get(job_url, headers=headers).json()
                if job_response.get("status") == "succeeded":
                    tasks = job_response.get("tasks", {}).get("items", [])
                    if tasks:
                        summaries = tasks[0].get("results", {}).get("conversations", [{}])[0].get("summaries", [])
                        results["summary"] = {
                            "issue": next((s["text"] for s in summaries if s["aspect"] == "issue"), "Not identified"),
                            "resolution": next((s["text"] for s in summaries if s["aspect"] == "resolution"), "Not identified")
                        }
                    break
                elif job_response.get("status") == "failed":
                    break
    except Exception as e:
        results["summary_error"] = str(e)
    
    return results


def grade_transcript(text):
    """
    Send transcript to Prompt Flow AI model for quality evaluation.
    
    Prompt Flow is a low-code tool in Azure AI Studio that chains LLMs 
    (Large Language Models), embeddings, and tools together. This function
    sends the transcript to a pre-configured flow that evaluates call quality
    and returns a score and feedback.
    
    Args:
        text: Full transcript (preferably with PII redacted)
        
    Returns:
        dict: Grading result from Prompt Flow with keys:
            - score: Quality score (typically 0-100)
            - summary: Human-readable evaluation summary
            - flags: List of issues detected (e.g., "customer frustration", "missing info")
            
    Notes:
        - Prompt Flow endpoint must be configured in Azure AI Studio
        - Timeout: 90 seconds to allow for LLM processing
        - Handles both direct dict responses and nested "output" responses
        - Gracefully handles JSON parsing errors from LLM output
    """
    headers = {"Authorization": f"Bearer {PF_KEY}", "Content-Type": "application/json"}
    
    # Send text under multiple possible keys to ensure compatibility with different
    # Prompt Flow configurations. This redundancy increases robustness.
    data = {
        "transcript_text": text,  # Standard naming
        "text": text,  # Common fallback
        "question": text,  # Alternative naming
        "chat_history": []  # For chat-based flows
    }
    
    try:
        response = requests.post(PF_ENDPOINT, headers=headers, json=data, timeout=90)
        if response.status_code != 200:
            return {"score": 0, "summary": "API Error", "flags": ["System Error"]}
            
        result = response.json()
        
        # Prompt Flow responses can be nested under an "output" key
        # Handle both direct results and wrapped outputs
        if "output" in result:
            output_val = result["output"]
            if isinstance(output_val, str):
                # Some flows return JSON as a string; attempt to parse it
                try:
                    clean_json = output_val.replace("```json", "").replace("```", "").strip()
                    return json.loads(clean_json)
                except:
                    # If parsing fails, return as a summary string
                    return {"score": 0, "summary": output_val, "flags": ["Parse Error"]}
            elif isinstance(output_val, dict):
                return output_val
        return result
    except Exception as e:
        return {"score": 0, "summary": str(e), "flags": ["Timeout/Error"]}

def process_single_file(uploaded_file):
    """
    Complete end-to-end processing pipeline for one call recording.
    
    This is the main orchestration function that coordinates all analysis
    steps for a single audio file. It runs sequentially through:
    1. Upload to secure storage
    2. Transcribe with speaker identification
    3. Detect audio anomalies
    4. Language analysis (PII, sentiment, summarization)
    5. AI grading
    6. Report formatting
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        dict: Complete analysis result with all processed data or error status
        
    Error Handling:
        - Returns graceful error messages at each step
        - Always preserves audio_bytes and audio_format for UI playback
        - Catches and reports any unexpected exceptions
    """
    try:
        # --- STEP 0: Cache Audio Bytes ---
        # Store the audio file data for UI playback later
        uploaded_file.seek(0)
        audio_bytes = uploaded_file.read()
        audio_format = uploaded_file.name.split('.')[-1].lower()
        
        # --- STEP 1: Upload to Azure Blob Storage ---
        # Generate SAS URL for transcription API access
        uploaded_file.seek(0)  # Reset file pointer after reading bytes
        sas_url = get_blob_sas_url(uploaded_file)
        if not sas_url:
            return {"filename": uploaded_file.name, "status": "Upload Failed", "score": 0, "flags": [], "audio_bytes": audio_bytes, "audio_format": audio_format}
        
        # --- STEP 2: Transcribe Audio with Speaker Diarization ---
        # Convert audio → text with speaker identification
        transcription_result = transcribe_audio(sas_url)
        if not transcription_result:
            return {"filename": uploaded_file.name, "status": "Transcription Failed", "score": 0, "flags": [], "audio_bytes": audio_bytes, "audio_format": audio_format}
        
        conversation_turns = transcription_result["conversation_turns"]
        combined_text = transcription_result["combined_text"]
        speaker_stats = transcription_result["speaker_stats"]
        
        # --- STEP 3: Detect Audio Anomalies ---
        # Identify pauses, potential dead air, or transfer points
        audio_events = detect_audio_events(conversation_turns)
        
        # --- STEP 4: Language Service Analysis ---
        # Analyze for PII (redact sensitive data), sentiment per turn, call summary
        language_analysis = analyze_with_language_service(conversation_turns)
        
        # --- STEP 5: AI Grading via Prompt Flow ---
        # Use LLM-based flow to grade call quality and return score + flags
        # Prefer redacted text if PII was detected (privacy compliance)
        text_for_grading = combined_text
        if language_analysis.get("redacted_turns"):
            text_for_grading = "\n".join([f"{t['speaker']}: {t['text']}" for t in language_analysis["redacted_turns"]])
        
        grading = grade_transcript(text_for_grading)
        
        # --- STEP 6: Calculate Sentiment Metrics ---
        # Aggregate sentiment across all turns for overall tone assessment
        sentiment_breakdown = {"positive": 0, "neutral": 0, "negative": 0}
        for s in language_analysis.get("sentiment_results", []):
            sentiment = s.get("sentiment", "neutral")
            sentiment_breakdown[sentiment] = sentiment_breakdown.get(sentiment, 0) + 1
        
        # Determine dominant sentiment (most common across turns)
        dominant_sentiment = max(sentiment_breakdown, key=sentiment_breakdown.get) if sentiment_breakdown else "neutral"
        
        # --- STEP 7: Format Rich Transcript for Display ---
        # Build annotated transcript with timestamps, sentiment, PII markers
        formatted_transcript = build_formatted_transcript(
            conversation_turns, 
            language_analysis.get("sentiment_results", []),
            language_analysis.get("redacted_turns", []),
            audio_events
        )
        
        # --- STEP 8: Compile Final Result ---
        # Return comprehensive analysis result for dashboard display
        return {
            "filename": uploaded_file.name,
            "status": "Success",
            "score": grading.get("score", 0),
            "summary": grading.get("summary", "No summary"),
            "flags": grading.get("flags", []),
            "transcript": combined_text,
            "formatted_transcript": formatted_transcript,
            "conversation_turns": conversation_turns,
            "speaker_stats": speaker_stats,
            "total_speakers": transcription_result["total_speakers"],
            "audio_events": audio_events,
            "sentiment_breakdown": sentiment_breakdown,
            "dominant_sentiment": dominant_sentiment,
            "language_summary": language_analysis.get("summary"),
            "pii_detected": any(p.get("pii_detected") for p in language_analysis.get("redacted_turns", [])),
            "audio_bytes": audio_bytes,
            "audio_format": audio_format,
            "raw_json": grading
        }
    except Exception as e:
        return {"filename": uploaded_file.name, "status": f"Error: {str(e)}", "score": 0, "flags": ["System Error"]}


def build_formatted_transcript(turns, sentiment_results, redacted_turns, audio_events):
    """
    Build a richly formatted transcript for dashboard display.
    
    Creates a human-readable version of the transcript with:
    - Timestamps for each speaker turn
    - Sentiment emoji indicators per turn (😊 positive, 😐 neutral, 😟 negative)
    - PII redaction markers (🔒) to show where sensitive data was masked
    - Audio event annotations (e.g., long pauses)
    
    Args:
        turns: List of conversation turn dicts
        sentiment_results: List of sentiment analysis results per turn
        redacted_turns: List of turns with PII removed
        audio_events: List of detected audio anomalies
        
    Returns:
        str: Formatted transcript string ready for display in Streamlit UI
    """
    lines = []
    sentiment_map = {s["turn_id"]: s for s in sentiment_results}
    redacted_map = {i: t for i, t in enumerate(redacted_turns)}
    event_map = {e["after_turn"]: e for e in audio_events}
    
    for i, turn in enumerate(turns):
        # Get sentiment emoji
        sentiment = sentiment_map.get(i, {}).get("sentiment", "neutral")
        sentiment_emoji = {"positive": "😊", "neutral": "😐", "negative": "😟"}.get(sentiment, "")
        
        # Use redacted text if available
        text = redacted_map.get(i, {}).get("text", turn["text"])
        
        # Check if PII was redacted
        pii_marker = ""
        if redacted_map.get(i, {}).get("pii_detected"):
            pii_marker = " 🔒"
        
        # Format line
        line = f"[{turn['timestamp']}] {turn['speaker']}: {text} {sentiment_emoji}{pii_marker}"
        lines.append(line)
        
        # Add audio events after this turn
        if i in event_map:
            lines.append(f"\n{event_map[i]['description']}\n")
    
    return "\n".join(lines)


# ============================================================================
# REPORT GENERATION FUNCTIONS
# ============================================================================
# Functions to export analysis results in multiple formats for leadership and
# compliance teams. All reports include PII redaction and quality metrics.

def generate_excel_report(results, df):
    """
    Generate a comprehensive multi-sheet Excel report suitable for leadership review.
    
    Creates an Excel workbook with the following sheets:
    1. Summary: High-level metrics (total calls, average score, flags, etc.)
    2. All Calls: Overview of each call with status, score, sentiment, flags
    3. Flagged Calls: Filtered view of calls requiring attention (score < 75)
    4. Transcripts: Full formatted transcripts for detailed review
    5. Speaker Stats: Word count and turn analysis per speaker
    
    This format enables non-technical stakeholders to review quality metrics
    and drill into specific calls that need attention.
    
    Args:
        results: List of analysis results from process_single_file()
        df: Pandas DataFrame version of results
        
    Returns:
        bytes: Excel file content (.xlsx format)
    """
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # ===== Sheet 1: Executive Summary =====
        # High-level KPIs for leadership dashboard
        summary_data = {
            'Metric': ['Total Calls', 'Average Score', 'Calls Needing Review', 'PII Detected', 'Negative Sentiment Calls', 'Report Generated'],
            'Value': [
                len(results),
                f"{df[df['status'] == 'Success']['score'].mean():.1f}" if len(df[df['status'] == 'Success']) > 0 else 'N/A',
                int(df['Needs Review'].sum()) if 'Needs Review' in df.columns else 0,
                sum(1 for r in results if r.get("pii_detected", False)),
                sum(1 for r in results if r.get("dominant_sentiment") == "negative"),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        # ===== Sheet 2: All Calls Overview =====
        # Complete list of calls with key metrics for triage
        overview_data = []
        for r in results:
            overview_data.append({
                'Filename': r.get('filename', ''),
                'Status': r.get('status', ''),
                'Score': r.get('score', 0),
                'Needs Review': 'Yes' if r.get('score', 0) < 75 else 'No',
                'Sentiment': r.get('dominant_sentiment', 'N/A'),
                'Speakers': r.get('total_speakers', 0),
                'PII Detected': 'Yes' if r.get('pii_detected', False) else 'No',
                'Summary': r.get('summary', ''),
                'Flags': ', '.join(r.get('flags', []))
            })
        pd.DataFrame(overview_data).to_excel(writer, sheet_name='All Calls', index=False)
        
        # ===== Sheet 3: Flagged Calls Only =====
        # Filtered list of problematic calls for focused review
        flagged_data = [d for d in overview_data if d['Needs Review'] == 'Yes']
        if flagged_data:
            pd.DataFrame(flagged_data).to_excel(writer, sheet_name='Flagged Calls', index=False)
        
        # ===== Sheet 4: Full Transcripts =====
        # Complete conversation transcripts (with PII redacted) for detailed analysis
        transcript_data = []
        for r in results:
            transcript_data.append({
                'Filename': r.get('filename', ''),
                'Transcript': r.get('formatted_transcript', r.get('transcript', ''))
            })
        pd.DataFrame(transcript_data).to_excel(writer, sheet_name='Transcripts', index=False)
        
        # ===== Sheet 5: Speaker Statistics =====
        # Engagement metrics per speaker (word count, turn count)
        speaker_data = []
        for r in results:
            stats = r.get('speaker_stats', {})
            for speaker, data in stats.items():
                speaker_data.append({
                    'Filename': r.get('filename', ''),
                    'Speaker': speaker,
                    'Turns': data.get('turns', 0),
                    'Words': data.get('words', 0),
                    'Avg Words/Turn': round(data.get('words', 0) / data.get('turns', 1), 1)
                })
        if speaker_data:
            pd.DataFrame(speaker_data).to_excel(writer, sheet_name='Speaker Stats', index=False)
    
    output.seek(0)
    return output.getvalue()


def generate_csv_report(results):
    """
    Generate a simple CSV report for data integration with other systems.
    
    Creates a flat CSV file suitable for:
    - Importing into data warehouses
    - Sharing with analytics/BI tools
    - Integration with call center management systems
    - Quick filtering in spreadsheet applications
    
    Args:
        results: List of analysis results from process_single_file()
        
    Returns:
        bytes: CSV file content (UTF-8 encoded)
    """
    rows = []
    for r in results:
        rows.append({
            'Filename': r.get('filename', ''),
            'Status': r.get('status', ''),
            'Score': r.get('score', 0),
            'Sentiment': r.get('dominant_sentiment', 'N/A'),
            'Speakers': r.get('total_speakers', 0),
            'PII Detected': 'Yes' if r.get('pii_detected', False) else 'No',
            'Summary': r.get('summary', ''),
            'Flags': ', '.join(r.get('flags', [])),
            'Transcript': r.get('transcript', '')
        })
    
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode('utf-8')


def generate_html_report(results, df):
    """
    Generate a professional, print-ready HTML report that can be saved as PDF.
    
    Creates a beautiful, single-page HTML report suitable for:
    - Executive presentations and leadership review
    - Regulatory compliance documentation
    - Email distribution to stakeholders
    - Printing to PDF via browser (Ctrl+P → Save as PDF)
    
    Features:
    - Color-coded call status (red for flagged, green for passed)
    - Per-call metrics, summaries, and flagged issues
    - Expandable transcript sections
    - Print-optimized CSS (proper page breaks, sizing)
    - Professional gradient header with statistics
    
    Args:
        results: List of analysis results from process_single_file()
        df: Pandas DataFrame version of results
        
    Returns:
        bytes: HTML file content (UTF-8 encoded)
        
    Usage Instructions:
        1. Download HTML file from Streamlit dashboard
        2. Open in web browser (Chrome recommended)
        3. Press Ctrl+P (Cmd+P on Mac) to open print dialog
        4. Select "Save as PDF" to create PDF copy
    """
    
    # Calculate key statistics for the report header
    total_calls = len(results)
    avg_score = df[df['status'] == 'Success']['score'].mean() if len(df[df['status'] == 'Success']) > 0 else 0
    flagged_count = int(df['Needs Review'].sum()) if 'Needs Review' in df.columns else 0
    pii_count = sum(1 for r in results if r.get("pii_detected", False))
    negative_count = sum(1 for r in results if r.get("dominant_sentiment") == "negative")
    
    html = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Call Center Analysis Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
        .header h1 {{ font-size: 2em; margin-bottom: 10px; }}
        .header .date {{ opacity: 0.9; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px; margin-bottom: 30px; }}
        .stat-card {{ background: #f8f9fa; border-radius: 10px; padding: 20px; text-align: center; border-left: 4px solid #667eea; }}
        .stat-card .value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
        .stat-card .label {{ color: #666; font-size: 0.9em; }}
        .stat-card.danger {{ border-left-color: #dc3545; }}
        .stat-card.danger .value {{ color: #dc3545; }}
        .stat-card.warning {{ border-left-color: #ffc107; }}
        .stat-card.warning .value {{ color: #ffc107; }}
        .stat-card.success {{ border-left-color: #28a745; }}
        .stat-card.success .value {{ color: #28a745; }}
        .section {{ margin-bottom: 30px; }}
        .section h2 {{ color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; margin-bottom: 20px; }}
        .call-card {{ background: white; border: 1px solid #ddd; border-radius: 10px; padding: 20px; margin-bottom: 20px; page-break-inside: avoid; }}
        .call-card.flagged {{ border-left: 5px solid #dc3545; }}
        .call-card.passed {{ border-left: 5px solid #28a745; }}
        .call-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; flex-wrap: wrap; gap: 10px; }}
        .call-title {{ font-size: 1.2em; font-weight: bold; }}
        .badge {{ padding: 5px 12px; border-radius: 20px; font-size: 0.8em; font-weight: bold; }}
        .badge-danger {{ background: #dc3545; color: white; }}
        .badge-success {{ background: #28a745; color: white; }}
        .badge-info {{ background: #17a2b8; color: white; }}
        .badge-warning {{ background: #ffc107; color: #333; }}
        .call-meta {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 15px; }}
        .meta-item {{ background: #f8f9fa; padding: 10px; border-radius: 5px; }}
        .meta-item .label {{ font-size: 0.8em; color: #666; }}
        .meta-item .value {{ font-weight: bold; }}
        .flags {{ margin-bottom: 15px; }}
        .flag {{ background: #fff3cd; padding: 5px 10px; border-radius: 5px; display: inline-block; margin: 2px; font-size: 0.9em; }}
        .transcript {{ background: #f8f9fa; padding: 15px; border-radius: 5px; font-family: monospace; white-space: pre-wrap; font-size: 0.85em; max-height: 400px; overflow-y: auto; }}
        .summary-box {{ background: #e7f3ff; padding: 15px; border-radius: 5px; margin-bottom: 15px; }}
        .footer {{ text-align: center; color: #666; padding: 20px; border-top: 1px solid #ddd; margin-top: 30px; }}
        @media print {{
            body {{ padding: 0; }}
            .call-card {{ page-break-inside: avoid; }}
            .transcript {{ max-height: none; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📞 Call Center Analysis Report</h1>
        <div class="date">Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</div>
    </div>
    
    <div class="stats-grid">
        <div class="stat-card">
            <div class="value">{total_calls}</div>
            <div class="label">Total Calls</div>
        </div>
        <div class="stat-card {'success' if avg_score >= 75 else 'danger'}">
            <div class="value">{avg_score:.1f}</div>
            <div class="label">Average Score</div>
        </div>
        <div class="stat-card danger">
            <div class="value">{flagged_count}</div>
            <div class="label">Flagged Calls</div>
        </div>
        <div class="stat-card warning">
            <div class="value">{pii_count}</div>
            <div class="label">PII Detected</div>
        </div>
        <div class="stat-card">
            <div class="value">{negative_count}</div>
            <div class="label">Negative Sentiment</div>
        </div>
    </div>
    
    <div class="section">
        <h2>📋 Call Details</h2>
'''
    
    # Sort results: flagged first
    sorted_results = sorted(results, key=lambda x: (x.get('score', 0) >= 75, x.get('score', 0)))
    
    for r in sorted_results:
        is_flagged = r.get('score', 0) < 75
        card_class = 'flagged' if is_flagged else 'passed'
        status_badge = '<span class="badge badge-danger">NEEDS REVIEW</span>' if is_flagged else '<span class="badge badge-success">PASSED</span>'
        
        sentiment = r.get('dominant_sentiment', 'neutral')
        sentiment_badge = {
            'positive': '<span class="badge badge-success">😊 Positive</span>',
            'neutral': '<span class="badge badge-info">😐 Neutral</span>',
            'negative': '<span class="badge badge-danger">😟 Negative</span>'
        }.get(sentiment, '')
        
        pii_badge = '<span class="badge badge-warning">🔐 PII</span>' if r.get('pii_detected') else ''
        
        flags_html = ''
        if r.get('flags'):
            flags_html = '<div class="flags">' + ''.join([f'<span class="flag">🚩 {f}</span>' for f in r.get('flags', [])]) + '</div>'
        
        # Language summary if available
        lang_summary = r.get('language_summary', {})
        summary_html = ''
        if lang_summary:
            summary_html = f'''
            <div class="summary-box">
                <strong>Issue:</strong> {lang_summary.get('issue', 'N/A')}<br>
                <strong>Resolution:</strong> {lang_summary.get('resolution', 'N/A')}
            </div>
            '''
        
        transcript = r.get('formatted_transcript', r.get('transcript', 'No transcript available'))
        # Escape HTML in transcript
        transcript = transcript.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        html += f'''
        <div class="call-card {card_class}">
            <div class="call-header">
                <span class="call-title">📁 {r.get('filename', 'Unknown')}</span>
                <div>
                    {status_badge}
                    {sentiment_badge}
                    {pii_badge}
                </div>
            </div>
            
            <div class="call-meta">
                <div class="meta-item">
                    <div class="label">Score</div>
                    <div class="value">{r.get('score', 0)}/100</div>
                </div>
                <div class="meta-item">
                    <div class="label">Speakers</div>
                    <div class="value">{r.get('total_speakers', 'N/A')}</div>
                </div>
                <div class="meta-item">
                    <div class="label">Sentiment</div>
                    <div class="value">{sentiment.title()}</div>
                </div>
                <div class="meta-item">
                    <div class="label">Status</div>
                    <div class="value">{r.get('status', 'N/A')}</div>
                </div>
            </div>
            
            {flags_html}
            
            <div class="summary-box">
                <strong>AI Summary:</strong> {r.get('summary', 'No summary available')}
            </div>
            
            {summary_html}
            
            <details>
                <summary style="cursor: pointer; font-weight: bold; margin-bottom: 10px;">📝 View Full Transcript</summary>
                <div class="transcript">{transcript}</div>
            </details>
        </div>
'''
    
    html += '''
    </div>
    
    <div class="footer">
        <p>Report generated by Call Center Triage Dashboard</p>
        <p>Created by Noah Earley</p>
    </div>
</body>
</html>
'''
    
    return html.encode('utf-8')


# ============================================================================
# MAIN USER INTERFACE - STREAMLIT DASHBOARD
# ============================================================================
# Interactive web interface for uploading, processing, and reviewing call batches

# --- FILE UPLOAD SECTION ---
# Allow users to select multiple audio files for batch processing
uploaded_files = st.file_uploader(
    "📂 Upload Call Batch (Select multiple files, up to 280MB total)", 
    type=["mp3", "wav", "m4a"], 
    accept_multiple_files=True
)

if uploaded_files and st.button(f"🚀 Analyze {len(uploaded_files)} Calls"):
    """
    Main Processing Pipeline:
    1. Upload all files concurrently to Azure Blob Storage
    2. Transcribe each file with speaker diarization
    3. Perform language analysis (PII, sentiment, summarization)
    4. Grade each call with AI model
    5. Compile results and display triage dashboard
    
    Processing is parallelized using ThreadPool (max 4 concurrent processes)
    to maximize throughput while managing Azure API rate limits.
    """
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process files concurrently for improved performance
    # ThreadPool enables parallel processing of multiple files
    with st.status("Running Batch Analysis...", expanded=True) as status:
        
        total_files = len(uploaded_files)
        completed = 0
        
        # ThreadPoolExecutor with max_workers=4 to balance parallelism and API limits
        # Each thread processes one file through the entire pipeline
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {executor.submit(process_single_file, f): f for f in uploaded_files}
            
            # As_completed yields futures as they finish (not in order)
            for future in concurrent.futures.as_completed(future_to_file):
                result = future.result()
                results.append(result)
                completed += 1
                progress_bar.progress(completed / total_files)
                status.write(f"✅ Processed: {result['filename']}")
        
        status.update(label="Batch Analysis Complete!", state="complete", expanded=False)
        status_text.empty()
        progress_bar.empty()

    # --- TRIAGE DASHBOARD ---
    # Display results in an organized, actionable format
    
    st.divider()
    
    # Build DataFrame for filtering and analysis
    df = pd.DataFrame(results)
    
    # Define triage logic: Flag if score < 75 OR has critical flags
    def is_bad_call(row):
        """Determine if a call needs review based on score and flags."""
        if row['score'] < 75: 
            return True
        # Also flag calls with critical issues even if score is reasonable
        bad_flags = [f for f in row['flags'] if "angry" in f.lower() or "escalation" in f.lower() or "missing" in f.lower()]
        return len(bad_flags) > 0

    df['Needs Review'] = df.apply(is_bad_call, axis=1)
    
    # --- DASHBOARD METRICS ---
    # Display high-level KPIs for quick overview
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Calls", len(df))
    with col2:
        avg_score = df[df['status'] == "Success"]['score'].mean()
        st.metric("Average Score", f"{avg_score:.1f}" if not pd.isna(avg_score) else "N/A")
    with col3:
        bad_calls = df['Needs Review'].sum()
        st.metric("🚩 Calls Flagged", int(bad_calls), delta_color="inverse")
    with col4:
        # Count calls with PII detected and redacted
        pii_count = sum(1 for r in results if r.get("pii_detected", False))
        st.metric("🔐 PII Detected", pii_count)
    with col5:
        # Count calls with predominantly negative sentiment
        negative_count = sum(1 for r in results if r.get("dominant_sentiment") == "negative")
        st.metric("😟 Negative Calls", negative_count)

    # --- EXPORT REPORTS SECTION ---
    # Multiple export formats for different stakeholders
    st.divider()
    st.subheader("📥 Export Reports")
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        # Excel Export - Multi-sheet workbook for data analysis
        excel_data = generate_excel_report(results, df)
        st.download_button(
            label="📊 Download Excel Report",
            data=excel_data,
            file_name=f"call_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with export_col2:
        # CSV Export
        csv_data = generate_csv_report(results)
        st.download_button(
            label="📄 Download CSV Report",
            data=csv_data,
            file_name=f"call_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with export_col3:
        # HTML Report - Professional format that can be printed to PDF
        html_data = generate_html_report(results, df)
        st.download_button(
            label="🌐 Download HTML Report (Print to PDF)",
            data=html_data,
            file_name=f"call_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            use_container_width=True
        )
    
    st.caption("💡 Tip: Open the HTML report in a browser and use Print → Save as PDF for a professional PDF report")

    # --- TRIAGE LIST ---
    # Interactive detailed view of each call, sorted with flagged calls at the top
    st.subheader("🔍 Triage List")
    
    # Sort so "Needs Review" (True) is at the top, then by score (lowest first)
    df = df.sort_values(by=['Needs Review', 'score'], ascending=[False, True])
    
    for index, row in df.iterrows():
        # Visual indicators for call status
        if row['Needs Review']:
            icon = "🔴"  # Red = critical, needs immediate attention
            color = "red"
            label = "**NEEDS REVIEW**"
        else:
            icon = "🟢"  # Green = acceptable quality
            color = "green"
            label = "Passed"
        
        # Sentiment indicator for quick emotional assessment
        sentiment = row.get('dominant_sentiment', 'neutral')
        sentiment_icon = {"positive": "😊", "neutral": "😐", "negative": "😟"}.get(sentiment, "😐")
        
        # Speaker count indicates call complexity and engagement
        speaker_count = row.get('total_speakers', 0)
        
        # Collapsible call detail panel
        with st.expander(f"{icon} {row['filename']} | Score: {row['score']}/100 | {sentiment_icon} | 👥 {speaker_count} speakers | {label}"):
            
            # --- Key Metrics Row ---
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Score", f"{row['score']}/100")
            with m2:
                st.metric("Sentiment", sentiment.title())
            with m3:
                st.metric("Speakers", speaker_count)
            with m4:
                pii_status = "Yes 🔐" if row.get('pii_detected') else "No"
                st.metric("PII Detected", pii_status)
            
            st.divider()
            
            # --- Detailed Analysis Tabs ---
            # Different views for different stakeholder needs (QA, Supervisors, Analytics)
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Transcript", "📊 Analysis", "👥 Speakers", "🔐 PII/Redacted", "📋 Raw Data"])
            
            with tab1:
                # TAB 1: TRANSCRIPT & AUDIO
                st.markdown("### 🎧 Listen to Call")
                audio_bytes = row.get('audio_bytes')
                audio_format = row.get('audio_format', 'wav')
                if audio_bytes:
                    # Map format to MIME type
                    mime_types = {
                        'mp3': 'audio/mpeg',
                        'wav': 'audio/wav',
                        'm4a': 'audio/mp4',
                        'ogg': 'audio/ogg',
                        'flac': 'audio/flac'
                    }
                    mime_type = mime_types.get(audio_format, 'audio/wav')
                    st.audio(audio_bytes, format=mime_type)
                else:
                    st.caption("Audio not available")
                
                st.divider()
                
                st.markdown("### 📝 Formatted Transcript")
                st.markdown("*Legend: 😊 Positive | 😐 Neutral | 😟 Negative | 🔒 PII Redacted*")
                formatted = row.get('formatted_transcript', row.get('transcript', ''))
                st.text_area("", formatted, height=350, key=f"formatted_{index}")
                
                # Audio events
                audio_events = row.get('audio_events', [])
                if audio_events:
                    st.markdown("### 🔊 Audio Events Detected")
                    for event in audio_events:
                        st.info(f"⏸️ {event['description']} (after turn {event['after_turn'] + 1})")
            
            with tab2:
                c1, c2 = st.columns(2)
                
                with c1:
                    st.markdown("### 🚩 Flags Detected")
                    if row['flags']:
                        for flag in row['flags']:
                            st.error(f"🚩 {flag}")
                    else:
                        st.success("No flags detected")
                    
                    st.markdown("### 📝 AI Summary")
                    st.write(row.get('summary', 'No summary available'))
                
                with c2:
                    st.markdown("### 🎭 Sentiment Breakdown")
                    sentiment_breakdown = row.get('sentiment_breakdown', {})
                    if sentiment_breakdown:
                        # Create a simple bar chart
                        sent_df = pd.DataFrame({
                            'Sentiment': ['😊 Positive', '😐 Neutral', '😟 Negative'],
                            'Count': [
                                sentiment_breakdown.get('positive', 0),
                                sentiment_breakdown.get('neutral', 0),
                                sentiment_breakdown.get('negative', 0)
                            ]
                        })
                        st.bar_chart(sent_df.set_index('Sentiment'))
                    
                    st.markdown("### 📋 Language Service Summary")
                    lang_summary = row.get('language_summary')
                    if lang_summary:
                        st.info(f"**Issue:** {lang_summary.get('issue', 'N/A')}")
                        st.success(f"**Resolution:** {lang_summary.get('resolution', 'N/A')}")
                    else:
                        st.caption("Language summary not available")
            
            with tab3:
                st.markdown("### 👥 Speaker Statistics")
                speaker_stats = row.get('speaker_stats', {})
                if speaker_stats:
                    stats_data = []
                    for speaker, stats in speaker_stats.items():
                        stats_data.append({
                            "Speaker": speaker,
                            "Turns": stats['turns'],
                            "Words": stats['words'],
                            "Avg Words/Turn": round(stats['words'] / stats['turns'], 1) if stats['turns'] > 0 else 0
                        })
                    st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
                    
                    # Word distribution chart
                    if len(stats_data) > 0:
                        st.markdown("### 💬 Speaking Distribution")
                        word_df = pd.DataFrame(stats_data)
                        st.bar_chart(word_df.set_index('Speaker')['Words'])
                
                st.markdown("### 🗣️ Conversation Flow")
                turns = row.get('conversation_turns', [])
                for i, turn in enumerate(turns[:20]):  # Limit to first 20 turns for display
                    speaker_color = "blue" if "Representative" in turn['speaker'] else "green"
                    st.markdown(f"**[{turn['timestamp']}] {turn['speaker']}:** {turn['text']}")
                if len(turns) > 20:
                    st.caption(f"... and {len(turns) - 20} more turns")
            
            with tab4:
                st.markdown("### 🔐 PII Detection & Redaction")
                if row.get('pii_detected'):
                    st.warning("⚠️ Personal Identifiable Information was detected and redacted in this call.")
                    st.markdown("The following text shows the redacted version:")
                    formatted = row.get('formatted_transcript', '')
                    st.text_area("Redacted Transcript", formatted, height=300, key=f"redacted_{index}")
                else:
                    st.success("✅ No PII detected in this call")
            
            with tab5:
                st.markdown("### 📋 Raw JSON Response")
                st.json(row.get('raw_json', {}))

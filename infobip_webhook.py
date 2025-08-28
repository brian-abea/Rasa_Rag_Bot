import logging
import os
import httpx
import tempfile
import asyncio
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from collections import deque
from typing import Dict, Any, Optional, Tuple
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import langdetect
from langdetect import detect_langs

# Configure logging to provide clear output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# Configuration for Faster Whisper Voice Notes
model_size = "large-v3"

# Initialize Deep Translator (more reliable than googletrans)
translator = GoogleTranslator()

# Language code mappings for better compatibility
LANGUAGE_MAPPINGS = {
    'sw': 'swahili',     # Swahili
    'ki': 'swahili',     # Alternative Swahili code
    'en': 'english',     # English
    'es': 'spanish',     # Spanish
    'fr': 'french',      # French
    'pt': 'portuguese',  # Portuguese
    'ar': 'arabic',      # Arabic
    'hi': 'hindi',       # Hindi
    'yo': 'yoruba',      # Yoruba
    'ha': 'hausa',       # Hausa
    'ig': 'igbo',        # Igbo
    'am': 'amharic',     # Amharic
    'om': 'oromo',       # Oromo
    'lg': 'luganda',     # Luganda
    'ln': 'lingala',     # Lingala 
    'rw': 'kinyarwanda', # Kinyarwanda 
    'luo': 'luo',        # Kikuyu 
    'nso': 'sepedi',     # Sothafrican Sotho 
    'ts': 'tsonga',      # Southafrican Tsonga 
}

# Supported languages for validation
SUPPORTED_LANGUAGES = {'en', 'sw', 'es', 'fr', 'pt', 'ar', 'hi', 'yo', 'ha', 'ig', 'am', 'rw', 'de', 'it', 'ru', 'ja', 'ko', 'zh', 'th', 'vi'}

# Configuration for language detection
LANGUAGE_CONFIDENCE_THRESHOLD = 0.7
MIN_WORDS_FOR_DETECTION = 3

def initialize_whisper_model():
    """Initialize Whisper model with GPU fallback to CPU."""
    import os
    
    # Check if CUDA libraries are available and working
    def test_cuda_availability():
        try:
            import torch
            if torch.cuda.is_available():
                # Test basic CUDA operation
                torch.cuda.current_device()
                return True
        except Exception:
            pass
        return False
    
    # Check for cuDNN specifically
    def check_cudnn():
        try:
            import torch
            return torch.backends.cudnn.is_available()
        except Exception:
            return False
    
    cuda_available = test_cuda_availability()
    cudnn_available = check_cudnn()
    
    logger.info(f"CUDA available: {cuda_available}, cuDNN available: {cudnn_available}")
    
    # If CUDA/cuDNN issues detected, force CPU usage
    if not cuda_available or not cudnn_available:
        logger.warning("CUDA/cuDNN issues detected, using CPU for stability")
        try:
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            logger.info("Whisper model loaded successfully on CPU (forced)")
            return model, "cpu"
        except Exception as cpu_error:
            logger.error(f"Failed to load Whisper model on CPU: {cpu_error}")
            raise
    
    # Try GPU if CUDA seems healthy
    try:
        # First try GPU with FP16
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        logger.info("Whisper model loaded successfully on GPU (FP16)")
        return model, "cuda"
    except Exception as gpu_error:
        logger.warning(f"GPU (FP16) failed: {gpu_error}")
        try:
            # Try GPU with INT8
            model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
            logger.info("Whisper model loaded successfully on GPU (INT8)")
            return model, "cuda"
        except Exception as gpu_int8_error:
            logger.warning(f"GPU (INT8) failed: {gpu_int8_error}")
            try:
                # Fallback to CPU
                model = WhisperModel(model_size, device="cpu", compute_type="int8")
                logger.info("Whisper model loaded successfully on CPU (fallback)")
                return model, "cpu"
            except Exception as cpu_error:
                logger.error(f"Failed to load Whisper model on CPU: {cpu_error}")
                raise

# Initialize Whisper model for voice transcription
logger.info(f"Loading Whisper model: {model_size}")
model, device_used = initialize_whisper_model()

# Verify and retrieve essential environment variables
RASA_CORE_WEBHOOK_ENDPOINT = os.getenv(
    "RASA_CORE_WEBHOOK_ENDPOINT", "http://localhost:5005/webhooks/rest/webhook"
)
INFOBIP_API_KEY = os.getenv("INFOBIP_API_KEY")
WHATSAPP_SENDER = os.getenv("WHATSAPP_SENDER")
INFOBIP_SEND_API = os.getenv("INFOBIP_SEND_API")
INFOBIP_MEDIA_API = os.getenv("INFOBIP_MEDIA_API")

if not all([RASA_CORE_WEBHOOK_ENDPOINT, INFOBIP_API_KEY, WHATSAPP_SENDER, INFOBIP_SEND_API]):
    logger.error("Missing one or more required environment variables.")
    raise ValueError("Missing required environment variables.")

# In-memory cache for message IDs and user language preferences
PROCESSED_MESSAGE_IDS = deque(maxlen=200)
USER_LANGUAGE_CACHE = {}  # Store user's preferred language

# Create temp directory for audio files
TEMP_AUDIO_DIR = Path(tempfile.gettempdir()) / "whatsapp_audio"
TEMP_AUDIO_DIR.mkdir(exist_ok=True)


def detect_language_with_confidence(text: str) -> Tuple[str, float]:
    """Detect language of text input with confidence score."""
    try:
        # Clean and prepare text
        clean_text = text.strip().lower()
        word_count = len(clean_text.split())
        
        # For very short texts, assume English to avoid false positives
        if word_count < MIN_WORDS_FOR_DETECTION or len(clean_text) < 10:
            logger.info(f"Text too short ({word_count} words), defaulting to English: '{text[:30]}...'")
            return 'en', 1.0
        
        # Detect language with confidence
        detected_lang = langdetect.detect(text)
        lang_probs = detect_langs(text)
        
        # Find confidence for the detected language
        confidence = 0.0
        for prob in lang_probs:
            if prob.lang == detected_lang:
                confidence = prob.prob
                break
        
        # Log detection results
        logger.info(f"Language detection: {detected_lang} (confidence: {confidence:.2f}) for: '{text[:50]}...'")
        
        # If confidence is low, default to English
        if confidence < LANGUAGE_CONFIDENCE_THRESHOLD:
            logger.info(f"Low confidence ({confidence:.2f}), defaulting to English")
            return 'en', confidence
            
        # Validate detected language is supported
        if detected_lang not in SUPPORTED_LANGUAGES:
            logger.info(f"Unsupported language {detected_lang}, defaulting to English")
            return 'en', confidence
            
        return detected_lang, confidence
        
    except Exception as e:
        logger.warning(f"Language detection failed: {e}, defaulting to English")
        return 'en', 0.0


def detect_language(text: str) -> str:
    """Detect language of text input (backward compatibility)."""
    detected_lang, _ = detect_language_with_confidence(text)
    return detected_lang


async def translate_text(text: str, target_lang: str, source_lang: str = 'auto') -> str:
    """Translate text between languages using deep-translator."""
    if not text or not text.strip():
        return text
    
    try:
        # Skip translation if source and target are the same
        if source_lang == target_lang:
            return text
        
        # Skip translation if target is English and we detect English
        if target_lang == 'en':
            detected_lang, confidence = detect_language_with_confidence(text)
            if detected_lang == 'en' and confidence > LANGUAGE_CONFIDENCE_THRESHOLD:
                return text
        
        # Validate target language is supported
        if target_lang not in SUPPORTED_LANGUAGES:
            logger.warning(f"Unsupported target language: {target_lang}, returning original text")
            return text
        
        # Run translation in thread pool since deep-translator is synchronous
        loop = asyncio.get_event_loop()
        
        def translate_sync():
            try:
                # Create translator instance for this translation
                if source_lang == 'auto':
                    trans = GoogleTranslator(target=target_lang)
                else:
                    trans = GoogleTranslator(source=source_lang, target=target_lang)
                
                result = trans.translate(text)
                return result
            except Exception as e:
                logger.error(f"Translation error: {e}")
                return text
        
        translated_text = await loop.run_in_executor(None, translate_sync)
        
        # Log successful translation
        if translated_text != text:
            logger.info(f"Translated ({source_lang}→{target_lang}): '{text[:30]}...' → '{translated_text[:30]}...'")
        
        return translated_text
        
    except Exception as e:
        logger.error(f"Translation failed: {e}, returning original text")
        return text


async def download_media_file(media_url: str, headers: Dict[str, str]) -> Optional[Path]:
    """Download media file from Infobip and return local file path."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(media_url, headers=headers)
            response.raise_for_status()
            
            # Create temporary file
            temp_file = TEMP_AUDIO_DIR / f"audio_{os.urandom(8).hex()}.ogg"
            
            # Write audio data to file
            with open(temp_file, "wb") as f:
                f.write(response.content)
            
            logger.info(f"Downloaded audio file: {temp_file}")
            return temp_file
            
    except Exception as e:
        logger.error(f"Failed to download media file: {e}")
        return None


async def transcribe_audio(audio_file_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Transcribe audio file using Whisper model and return transcription + detected language."""
    try:
        logger.info(f"Transcribing audio file: {audio_file_path}")
        
        # Run transcription in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        def transcribe_sync():
            try:
                # Add timeout and error handling
                segments, info = model.transcribe(
                    str(audio_file_path), 
                    beam_size=5,
                    word_timestamps=False,  # Disable to reduce memory usage
                    vad_filter=True,       # Enable voice activity detection
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                return segments, info
            except Exception as e:
                error_msg = str(e).lower()
                logger.error(f"Transcription error: {e}")
                
                # Check for CUDA-related errors
                if any(keyword in error_msg for keyword in ['cuda', 'cudnn', 'gpu', 'device']):
                    logger.warning("GPU-related error detected, creating CPU model for this transcription...")
                    try:
                        # Create a temporary CPU model for this transcription
                        cpu_model = WhisperModel(model_size, device="cpu", compute_type="int8")
                        segments, info = cpu_model.transcribe(
                            str(audio_file_path), 
                            beam_size=5,
                            word_timestamps=False,
                            vad_filter=True,
                            vad_parameters=dict(min_silence_duration_ms=500)
                        )
                        logger.info("CPU fallback transcription successful")
                        return segments, info
                    except Exception as cpu_e:
                        logger.error(f"CPU fallback also failed: {cpu_e}")
                        raise cpu_e
                raise e
        
        segments, info = await loop.run_in_executor(None, transcribe_sync)
        
        # Extract transcribed text
        transcribed_text = " ".join([segment.text.strip() for segment in segments])
        detected_language = info.language
        
        logger.info(f"Detected language: {detected_language} (confidence: {info.language_probability:.2f})")
        logger.info(f"Transcription: {transcribed_text}")
        
        # Clean up temporary file
        try:
            audio_file_path.unlink()
            logger.info(f"Cleaned up temp file: {audio_file_path}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup temp file: {cleanup_error}")
        
        return (transcribed_text.strip() if transcribed_text.strip() else None, detected_language)
        
    except Exception as e:
        logger.error(f"Audio transcription failed: {e}")
        # Clean up file even on error
        try:
            if audio_file_path.exists():
                audio_file_path.unlink()
        except:
            pass
        return None, None


def get_fallback_message(original_language: str = 'en') -> str:
    """Get appropriate fallback message based on language."""
    fallback_messages = {
        'en': "Sorry, I didn't understand that. Could you please rephrase your question? I can help you with payments, asset status, smartphone financing, and general Watu services.",
        'sw': "Samahani, sijaeleweka. Je, unaweza kuuliza kwa njia nyingine? Ninaweza kukusaidia na malipo, hali ya mali, fedha za simu, na huduma za Watu.",
        'es': "Lo siento, no entendí eso. ¿Podrías reformular tu pregunta? Te puedo ayudar con pagos, estado de activos, financiamiento de smartphones y servicios generales de Watu.",
    }
    return fallback_messages.get(original_language, fallback_messages['en'])


async def process_with_rasa(message: str, sender: str, original_language: str = 'en') -> list:
    """Process message with Rasa and handle translation."""
    try:
        # Prepare message for Rasa (always in English)
        english_message = message
        
        # Only translate to English if we're confident it's not already English
        if original_language != 'en':
            # Double-check language detection for text messages
            detected_lang, confidence = detect_language_with_confidence(message)
            
            if detected_lang == 'en' and confidence > LANGUAGE_CONFIDENCE_THRESHOLD:
                # Actually English, update the original language
                original_language = 'en'
                logger.info(f"Text re-detected as English, no translation needed: {message}")
            else:
                # Translate to English for Rasa processing
                english_message = await translate_text(message, 'en', original_language)
                logger.info(f"Sending to Rasa (EN): {english_message}")
        else:
            logger.info(f"Sending to Rasa: {message}")
        
        # Send to Rasa
        rasa_payload = {"sender": sender, "message": english_message}
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            rasa_response = await client.post(RASA_CORE_WEBHOOK_ENDPOINT, json=rasa_payload)
            rasa_response.raise_for_status()
            bot_replies = rasa_response.json()
            
            logger.info(f"Rasa replied: {bot_replies}")
            
            # Handle empty Rasa responses
            if not bot_replies or len(bot_replies) == 0:
                logger.warning(f"Rasa returned empty response for message: '{message}'")
                fallback_text = get_fallback_message(original_language)
                logger.info(f"Sending fallback message in {original_language}: {fallback_text}")
                return [{"text": fallback_text}]
            
            # Translate responses back to original language if needed
            if original_language != 'en' and bot_replies:
                translated_replies = []
                for reply in bot_replies:
                    if 'text' in reply:
                        translated_text = await translate_text(reply['text'], original_language, 'en')
                        translated_reply = reply.copy()
                        translated_reply['text'] = translated_text
                        translated_replies.append(translated_reply)
                        logger.info(f"Translated response to {original_language}: {translated_text}")
                    else:
                        translated_replies.append(reply)
                return translated_replies
            
            return bot_replies
            
    except Exception as e:
        logger.error(f"Error processing with Rasa: {e}")
        # Return error message in user's language
        error_message = "Sorry, I'm having trouble processing your message right now. Please try again."
        if original_language != 'en':
            error_message = await translate_text(error_message, original_language, 'en')
        return [{"text": error_message}]


@app.post("/infobip-webhook")
async def infobip_to_rasa(request: Request):
    try:
        payload = await request.json()
        logger.info(f"Received raw Infobip payload: {payload}")

        # Ignore "results" payloads completely (delivery reports)
        if "results" in payload:
            logger.info("Skipping results[] payload (delivery report / duplicate inbound).")
            return {"status": "ignored"}

        sender_whatsapp_number = None
        user_message_text = None
        message_id = None
        media_url = None
        media_type = None
        detected_language = 'en'  # Default language

        # Handle only real INBOUND messages
        if payload.get("direction") == "INBOUND":
            sender_whatsapp_number = (
                payload.get("from")
                or payload.get("singleSendMessage", {}).get("from", {}).get("phoneNumber")
            )
            
            # Check for text message
            content = payload.get("content", {})
            user_message_text = content.get("text")
            
            # Check for voice/audio message
            if content.get("type") == "VOICE":
                media_url = content.get("mediaUrl") or content.get("url")
                media_type = "voice"
                logger.info(f"Voice message detected from {sender_whatsapp_number}")
            elif content.get("type") == "AUDIO" or payload.get("contentType") == "AUDIO":
                media_url = content.get("mediaUrl") or content.get("url")
                media_type = "audio"
                logger.info(f"Audio message detected from {sender_whatsapp_number}")
            
            message_id = payload.get("id")

        elif "singleSendMessage" in payload:  # fallback
            message_data = payload["singleSendMessage"]
            sender_whatsapp_number = message_data.get("from", {}).get("phoneNumber")
            content = message_data.get("content", {})
            user_message_text = content.get("text")
            
            # Check for voice/audio in fallback
            if content.get("type") in ["VOICE", "AUDIO"]:
                media_url = content.get("mediaUrl") or content.get("url")
                media_type = content.get("type").lower()
            
            message_id = payload.get("id")

        # Deduplication
        if message_id in PROCESSED_MESSAGE_IDS:
            logger.info(f"Duplicate message with ID {message_id} received. Skipping.")
            return {"status": "duplicate_message"}
        if message_id:
            PROCESSED_MESSAGE_IDS.append(message_id)

        # Process voice/audio messages
        if media_url and media_type in ["voice", "audio"]:
            logger.info(f"Processing {media_type} message from {sender_whatsapp_number}")
            logger.info(f"Media URL: {media_url}")
            
            # Set up headers for media download
            headers = {
                "Authorization": f"App {INFOBIP_API_KEY}",
                "Accept": "application/json",
            }
            
            # Download and transcribe audio
            audio_file_path = await download_media_file(media_url, headers)
            if audio_file_path:
                transcription_result = await transcribe_audio(audio_file_path)
                user_message_text, voice_language = transcription_result
                
                if user_message_text:
                    # Map Whisper language codes to supported codes
                    detected_language = voice_language if voice_language in SUPPORTED_LANGUAGES else 'en'
                    
                    # Store user's language preference
                    USER_LANGUAGE_CACHE[sender_whatsapp_number] = detected_language
                    
                    logger.info(f"Voice transcribed ({detected_language}): '{user_message_text}'")
                else:
                    user_message_text = "Sorry, I couldn't understand the voice message. Could you please type your message or try again?"
                    # Try to respond in user's cached language
                    cached_lang = USER_LANGUAGE_CACHE.get(sender_whatsapp_number, 'en')
                    if cached_lang != 'en':
                        user_message_text = await translate_text(user_message_text, cached_lang, 'en')
                    logger.warning("Voice transcription failed or empty")
            else:
                user_message_text = "Sorry, I couldn't process your voice message. Could you please try again or send a text message?"
                # Try to respond in user's cached language
                cached_lang = USER_LANGUAGE_CACHE.get(sender_whatsapp_number, 'en')
                if cached_lang != 'en':
                    user_message_text = await translate_text(user_message_text, cached_lang, 'en')
                logger.error("Failed to download voice message")

        # For text messages, detect language with improved confidence
        elif user_message_text:
            detected_language, confidence = detect_language_with_confidence(user_message_text)
            # Store user's language preference
            USER_LANGUAGE_CACHE[sender_whatsapp_number] = detected_language
            logger.info(f"Text message detected in {detected_language} (confidence: {confidence:.2f}): {user_message_text}")

        # Validate we have required data
        if not user_message_text or not sender_whatsapp_number:
            logger.warning(f"Could not extract sender or message. Skipping. Payload: {payload}")
            return {"status": "unsupported_payload_format"}

        logger.info(f"Processing message from {sender_whatsapp_number} in {detected_language}: {user_message_text}")

        # Process with Rasa (handles translation internally)
        bot_replies = await process_with_rasa(
            user_message_text, 
            sender_whatsapp_number.lstrip('+'), 
            detected_language
        )

        # Send replies back to user
        send_headers = {
            "Authorization": f"App {INFOBIP_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            for reply in bot_replies:
                if "text" in reply:
                    outbound_payload = {
                        "from": WHATSAPP_SENDER,
                        "to": sender_whatsapp_number,
                        "content": {"text": reply["text"]},
                    }
                    logger.info(f"Sending to Infobip: {outbound_payload}")
                    
                    send_resp = await client.post(INFOBIP_SEND_API, json=outbound_payload, headers=send_headers)
                    send_resp.raise_for_status()
                    logger.info(f"Infobip response: {send_resp.json()}")

    except Exception as e:
        logger.error(f"Error processing Infobip webhook payload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

    return {"status": "processed_and_replied"}


@app.get("/health")
async def health_check():
    """Health check endpoint to verify the service is running."""
    return {
        "status": "healthy", 
        "whisper_model": model_size,
        "device_used": device_used,
        "translator_available": translator is not None,
        "supported_languages": len(SUPPORTED_LANGUAGES),
        "temp_audio_dir": str(TEMP_AUDIO_DIR),
        "cached_user_languages": len(USER_LANGUAGE_CACHE),
        "confidence_threshold": LANGUAGE_CONFIDENCE_THRESHOLD
    }


@app.get("/languages")
async def supported_languages():
    """Get list of supported languages."""
    return {
        "supported_languages": LANGUAGE_MAPPINGS,
        "total_supported": len(SUPPORTED_LANGUAGES),
        "confidence_threshold": LANGUAGE_CONFIDENCE_THRESHOLD,
        "min_words_for_detection": MIN_WORDS_FOR_DETECTION
    }


@app.post("/clear-language-cache")
async def clear_language_cache():
    """Clear user language preferences cache."""
    USER_LANGUAGE_CACHE.clear()
    return {"status": "Language cache cleared", "remaining_entries": len(USER_LANGUAGE_CACHE)}


@app.post("/set-user-language")
async def set_user_language(user_number: str, language: str):
    """Manually set user's language preference for testing."""
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")
    
    USER_LANGUAGE_CACHE[user_number] = language
    return {"status": "success", "user": user_number, "language": language}


@app.on_event("startup")
async def startup_event():
    """Log startup information and test translator."""
    logger.info("Initializing Deep Translator...")
    try:
        # Test translator with a simple translation in thread pool
        loop = asyncio.get_event_loop()
        test_translation = await loop.run_in_executor(
            None, 
            lambda: GoogleTranslator(source='en', target='es').translate("Hello")
        )
        logger.info(f"Deep Translator initialized successfully: Hello -> {test_translation}")
    except Exception as e:
        logger.error(f"Failed to initialize Deep Translator: {e}")
        raise e
    
    logger.info("Multilingual WhatsApp Voice & Text Bot started successfully!")
    logger.info(f"Whisper model: {model_size} on {device_used}")
    logger.info(f"Deep Translator: Available")
    logger.info(f"Supported languages: {len(SUPPORTED_LANGUAGES)}")
    logger.info(f"Temp audio directory: {TEMP_AUDIO_DIR}")
    logger.info(f"Rasa endpoint: {RASA_CORE_WEBHOOK_ENDPOINT}")
    logger.info(f"Language confidence threshold: {LANGUAGE_CONFIDENCE_THRESHOLD}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Watu Voice & Text Assistant")
    
    # Clean up any remaining temp files
    try:
        for temp_file in TEMP_AUDIO_DIR.glob("audio_*.ogg"):
            temp_file.unlink()
            logger.info(f"Cleaned up temp file: {temp_file}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp files on shutdown: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
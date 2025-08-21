import logging
import os
import httpx
import tempfile
import asyncio
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from collections import deque
from typing import Dict, Any, Optional
from faster_whisper import WhisperModel

# Configure logging to provide clear output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# Configuration for Faster Whisper Voice Notes
model_size = "large-v3"

# Initialize Whisper model for voice transcription
logger.info(f"Loading Whisper model: {model_size}")

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
            logger.info("✅ Whisper model loaded successfully on CPU (forced)")
            return model, "cpu"
        except Exception as cpu_error:
            logger.error(f"Failed to load Whisper model on CPU: {cpu_error}")
            raise
    
    # Try GPU if CUDA seems healthy
    try:
        # First try GPU with FP16
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        logger.info("✅ Whisper model loaded successfully on GPU (FP16)")
        return model, "cuda"
    except Exception as gpu_error:
        logger.warning(f"GPU (FP16) failed: {gpu_error}")
        try:
            # Try GPU with INT8
            model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
            logger.info("✅ Whisper model loaded successfully on GPU (INT8)")
            return model, "cuda"
        except Exception as gpu_int8_error:
            logger.warning(f"GPU (INT8) failed: {gpu_int8_error}")
            try:
                # Fallback to CPU
                model = WhisperModel(model_size, device="cpu", compute_type="int8")
                logger.info("✅ Whisper model loaded successfully on CPU (fallback)")
                return model, "cpu"
            except Exception as cpu_error:
                logger.error(f"Failed to load Whisper model on CPU: {cpu_error}")
                raise

model, device_used = initialize_whisper_model()

# Verify and retrieve essential environment variables
RASA_CORE_WEBHOOK_ENDPOINT = os.getenv(
    "RASA_CORE_WEBHOOK_ENDPOINT", "http://localhost:5005/webhooks/rest/webhook"
)
INFOBIP_API_KEY = os.getenv("INFOBIP_API_KEY")
WHATSAPP_SENDER = os.getenv("WHATSAPP_SENDER")
INFOBIP_SEND_API = os.getenv("INFOBIP_SEND_API")
INFOBIP_MEDIA_API = os.getenv("INFOBIP_MEDIA_API", "https://api.infobip.com/whatsapp/1/senders/{sender}/media")

if not all([RASA_CORE_WEBHOOK_ENDPOINT, INFOBIP_API_KEY, WHATSAPP_SENDER, INFOBIP_SEND_API]):
    logger.error("Missing one or more required environment variables.")
    raise ValueError("Missing required environment variables.")

# In-memory cache for message IDs to prevent duplicates
PROCESSED_MESSAGE_IDS = deque(maxlen=200)

# Create temp directory for audio files
TEMP_AUDIO_DIR = Path(tempfile.gettempdir()) / "whatsapp_audio"
TEMP_AUDIO_DIR.mkdir(exist_ok=True)


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
            
            logger.info(f"✅ Downloaded audio file: {temp_file}")
            return temp_file
            
    except Exception as e:
        logger.error(f"❌ Failed to download media file: {e}")
        return None


async def transcribe_audio(audio_file_path: Path) -> Optional[str]:
    """Transcribe audio file using Whisper model."""
    try:
        logger.info(f"🎙️ Transcribing audio file: {audio_file_path}")
        
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
                        logger.info("✅ CPU fallback transcription successful")
                        return segments, info
                    except Exception as cpu_e:
                        logger.error(f"CPU fallback also failed: {cpu_e}")
                        raise cpu_e
                raise e
        
        segments, info = await loop.run_in_executor(None, transcribe_sync)
        
        # Extract transcribed text
        transcribed_text = " ".join([segment.text.strip() for segment in segments])
        
        logger.info(f"🎯 Detected language: {info.language} (confidence: {info.language_probability:.2f})")
        logger.info(f"📝 Transcription: {transcribed_text}")
        
        # Clean up temporary file
        try:
            audio_file_path.unlink()
            logger.info(f"🗑️ Cleaned up temp file: {audio_file_path}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup temp file: {cleanup_error}")
        
        return transcribed_text.strip() if transcribed_text.strip() else None
        
    except Exception as e:
        logger.error(f"❌ Audio transcription failed: {e}")
        # Clean up file even on error
        try:
            if audio_file_path.exists():
                audio_file_path.unlink()
        except:
            pass
        return None


@app.post("/infobip-webhook")
async def infobip_to_rasa(request: Request):
    try:
        payload = await request.json()
        logger.info(f"📨 Received raw Infobip payload: {payload}")

        # 🚨 Ignore "results" payloads completely (delivery reports)
        if "results" in payload:
            logger.info("⏭️ Skipping results[] payload (delivery report / duplicate inbound).")
            return {"status": "ignored"}

        sender_whatsapp_number = None
        user_message_text = None
        message_id = None
        media_url = None
        media_type = None

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
                logger.info(f"🎙️ Voice message detected from {sender_whatsapp_number}")
            elif content.get("type") == "AUDIO" or payload.get("contentType") == "AUDIO":
                media_url = content.get("mediaUrl") or content.get("url")
                media_type = "audio"
                logger.info(f"🎵 Audio message detected from {sender_whatsapp_number}")
            
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
            logger.info(f"🔄 Duplicate message with ID {message_id} received. Skipping.")
            return {"status": "duplicate_message"}
        if message_id:
            PROCESSED_MESSAGE_IDS.append(message_id)

        # Process voice/audio messages
        if media_url and media_type in ["voice", "audio"]:
            logger.info(f"🎙️ Processing {media_type} message from {sender_whatsapp_number}")
            logger.info(f"📎 Media URL: {media_url}")
            
            # Set up headers for media download
            headers = {
                "Authorization": f"App {INFOBIP_API_KEY}",
                "Accept": "application/json",
            }
            
            # Download and transcribe audio
            audio_file_path = await download_media_file(media_url, headers)
            if audio_file_path:
                user_message_text = await transcribe_audio(audio_file_path)
                
                if user_message_text:
                    logger.info(f"✅ Voice transcribed: '{user_message_text}'")
                else:
                    user_message_text = "Sorry, I couldn't understand the voice message. Could you please type your message or try again?"
                    logger.warning("❌ Voice transcription failed or empty")
            else:
                user_message_text = "Sorry, I couldn't process your voice message. Could you please try again or send a text message?"
                logger.error("❌ Failed to download voice message")

        # Validate we have required data
        if not user_message_text or not sender_whatsapp_number:
            logger.warning(f"⚠️ Could not extract sender or message. Skipping. Payload: {payload}")
            return {"status": "unsupported_payload_format"}

        logger.info(f"✅ Processing message from {sender_whatsapp_number}: {user_message_text}")

        # Forward to Rasa
        rasa_payload = {"sender": sender_whatsapp_number.lstrip('+'), "message": user_message_text}
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            rasa_response = await client.post(RASA_CORE_WEBHOOK_ENDPOINT, json=rasa_payload)
            rasa_response.raise_for_status()
            bot_replies = rasa_response.json()
            logger.info(f"🤖 Rasa replied: {bot_replies}")

            # Prepare headers for sending messages
            send_headers = {
                "Authorization": f"App {INFOBIP_API_KEY}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

            # Send replies back to user
            for reply in bot_replies:
                if "text" in reply:
                    outbound_payload = {
                        "from": WHATSAPP_SENDER,
                        "to": sender_whatsapp_number,
                        "content": {"text": reply["text"]},
                    }
                    logger.info(f"📤 Sending to Infobip: {outbound_payload}")
                    
                    send_resp = await client.post(INFOBIP_SEND_API, json=outbound_payload, headers=send_headers)
                    send_resp.raise_for_status()
                    logger.info(f"✅ Infobip response: {send_resp.json()}")

    except Exception as e:
        logger.error(f"❌ Error processing Infobip webhook payload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

    return {"status": "processed_and_replied"}


@app.get("/health")
async def health_check():
    """Health check endpoint to verify the service is running."""
    return {
        "status": "healthy", 
        "whisper_model": model_size,
        "device_used": device_used,
        "temp_audio_dir": str(TEMP_AUDIO_DIR)
    }


@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    logger.info("🚀 WhatsApp Voice & Text Bot started successfully!")
    logger.info(f"🎙️ Whisper model: {model_size} on {device_used}")
    logger.info(f"📁 Temp audio directory: {TEMP_AUDIO_DIR}")
    logger.info(f"🤖 Rasa endpoint: {RASA_CORE_WEBHOOK_ENDPOINT}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("🛑 Shutting down WhatsApp Voice & Text Bot")
    
    # Clean up any remaining temp files
    try:
        for temp_file in TEMP_AUDIO_DIR.glob("audio_*.ogg"):
            temp_file.unlink()
            logger.info(f"🗑️ Cleaned up temp file: {temp_file}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp files on shutdown: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
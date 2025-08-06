#!/usr/bin/env python3
"""
AI Pipeline Setup Script

This script helps new users get the AI pipeline running quickly by:
1. Checking system requirements
2. Installing dependencies
3. Setting up models
4. Verifying the complete pipeline

Usage:
    python setup.py --help
    python setup.py --quick-start
    python setup.py --verify-only
"""

import argparse
import os
import sys
import subprocess
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class AISetupManager:
    """Manages the setup process for the AI pipeline"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.models_dir = self.project_root / "models"
        self.required_models = [
            "whisper",
            "translation", 
            "ner",
            "classifier",
            "summarization",
            "all_qa_distilbert_v1"
        ]
        
    def check_system_requirements(self) -> bool:
        """Check if system meets minimum requirements"""
        logger.info("🔍 Checking system requirements...")
        
        # Check Python version
        if sys.version_info < (3, 11):
            logger.error("❌ Python 3.11+ required. Current version: %s", sys.version)
            return False
        logger.info("✅ Python version: %s", sys.version.split()[0])
        
        # Check available disk space
        import shutil
        free_space_gb = shutil.disk_usage(self.project_root).free / (1024**3)
        if free_space_gb < 10:
            logger.warning("⚠️ Low disk space: %.1f GB (recommend 10+ GB)", free_space_gb)
        else:
            logger.info("✅ Disk space: %.1f GB available", free_space_gb)
            
        # Check for CUDA
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✅ CUDA GPU detected")
                return True
            else:
                logger.info("ℹ️ No CUDA GPU detected - will use CPU (slower)")
        except FileNotFoundError:
            logger.info("ℹ️ nvidia-smi not found - will use CPU")
            
        return True
    
    def install_dependencies(self) -> bool:
        """Install Python dependencies"""
        logger.info("📦 Installing Python dependencies...")
        
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            logger.error("❌ requirements.txt not found")
            return False
            
        try:
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error("❌ Failed to install dependencies: %s", e.stderr)
            return False
    
    def setup_model_directories(self) -> bool:
        """Create model directories if they don't exist"""
        logger.info("📁 Setting up model directories...")
        
        try:
            for model_name in self.required_models:
                model_dir = self.models_dir / model_name
                model_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"✅ Created directory: {model_dir}")
            return True
        except Exception as e:
            logger.error("❌ Failed to create model directories: %s", e)
            return False
    
    def setup_whisper_model(self) -> bool:
        """Set up faster-whisper model"""
        logger.info("🚀 Setting up Faster-Whisper model...")
        
        try:
            # Import and load the model
            sys.path.append(str(self.project_root))
            from app.models.whisper_model import WhisperModel
            
            model = WhisperModel()
            logger.info("📥 Loading/downloading Faster-Whisper model (this may take several minutes)...")
            
            success = model.load()
            if success:
                logger.info("✅ Faster-Whisper model ready!")
                
                # Quick test
                import numpy as np
                test_audio = np.zeros(16000, dtype=np.int16).tobytes()  # 1 second silence
                transcript = model.transcribe_pcm_audio(test_audio, sample_rate=16000, language='en')
                logger.info("✅ Model test completed")
                return True
            else:
                logger.error("❌ Failed to load Whisper model: %s", model.error)
                return False
                
        except Exception as e:
            logger.error("❌ Error setting up Whisper model: %s", e)
            return False
    
    def verify_all_models(self) -> Dict[str, bool]:
        """Verify all models can be loaded"""
        logger.info("🔍 Verifying all models...")
        
        model_status = {}
        
        try:
            sys.path.append(str(self.project_root))
            from app.models.model_loader import ModelLoader
            
            loader = ModelLoader()
            
            # Try to load each model
            for model_name in self.required_models:
                try:
                    if model_name == "whisper":
                        from app.models.whisper_model import WhisperModel
                        model = WhisperModel()
                        success = model.load()
                    elif model_name == "translation":
                        from app.models.translator_model import TranslatorModel
                        model = TranslatorModel()
                        success = model.load()
                    elif model_name == "ner":
                        from app.models.ner_model import NERModel
                        model = NERModel()
                        success = model.load()
                    elif model_name == "classifier":
                        from app.models.classifier_model import ClassifierModel
                        model = ClassifierModel()
                        success = model.load()
                    elif model_name == "summarization":
                        from app.models.summarizer_model import SummarizerModel
                        model = SummarizerModel()
                        success = model.load()
                    elif model_name == "all_qa_distilbert_v1":
                        from app.models.qa_model import QAModel
                        model = QAModel()
                        success = model.load()
                    else:
                        success = False
                    
                    model_status[model_name] = success
                    status_icon = "✅" if success else "❌"
                    logger.info(f"{status_icon} {model_name}: {'Ready' if success else 'Failed'}")
                    
                except Exception as e:
                    model_status[model_name] = False
                    logger.error(f"❌ {model_name}: Error - {e}")
            
            return model_status
            
        except Exception as e:
            logger.error("❌ Failed to verify models: %s", e)
            return {model: False for model in self.required_models}
    
    def test_complete_pipeline(self) -> bool:
        """Test the complete audio processing pipeline"""
        logger.info("🧪 Testing complete pipeline...")
        
        try:
            sys.path.append(str(self.project_root))
            
            # Create test audio (3 seconds of sine wave)
            import numpy as np
            sample_rate = 16000
            duration = 3.0
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.3
            audio_bytes = (audio * 32767).astype(np.int16).tobytes()
            
            # Test transcription
            from app.models.whisper_model import WhisperModel
            whisper = WhisperModel()
            whisper.load()
            
            start_time = time.time()
            transcript = whisper.transcribe_pcm_audio(audio_bytes, sample_rate=16000, language='en')
            processing_time = time.time() - start_time
            
            logger.info(f"✅ Pipeline test completed in {processing_time:.3f} seconds")
            logger.info(f"📝 Test result: '{transcript}' ({len(transcript)} chars)")
            
            return True
            
        except Exception as e:
            logger.error("❌ Pipeline test failed: %s", e)
            return False
    
    def create_env_file(self) -> bool:
        """Create .env file from template if it doesn't exist"""
        env_file = self.project_root / ".env"
        env_example = self.project_root / ".env.example"
        
        if env_file.exists():
            logger.info("✅ .env file already exists")
            return True
        
        if env_example.exists():
            try:
                import shutil
                shutil.copy2(env_example, env_file)
                logger.info("✅ Created .env file from template")
                return True
            except Exception as e:
                logger.error("❌ Failed to create .env file: %s", e)
                return False
        else:
            logger.warning("⚠️ No .env.example found, skipping .env creation")
            return True
    
    def print_next_steps(self):
        """Print next steps for the user"""
        logger.info("🎉 Setup completed successfully!")
        print("\n" + "="*60)
        print("🚀 NEXT STEPS")
        print("="*60)
        print()
        print("1. Start the services:")
        print("   docker-compose up -d")
        print()
        print("2. Check system health:")
        print("   curl http://localhost:8123/health/detailed")
        print()
        print("3. Test with an audio file:")
        print("   curl -X POST -F 'audio=@sample.wav' -F 'language=sw' \\")
        print("        http://localhost:8123/audio/process")
        print()
        print("4. Access API documentation:")
        print("   http://localhost:8123/docs")
        print()
        print("📚 For more information, see:")
        print("   - README.md")
        print("   - docs/model-setup-guide.md")
        print()

def main():
    parser = argparse.ArgumentParser(description="AI Pipeline Setup Script")
    parser.add_argument('--quick-start', action='store_true', 
                       help='Run complete setup process')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify existing installation')
    parser.add_argument('--models-only', action='store_true',
                       help='Only set up models')
    parser.add_argument('--skip-deps', action='store_true',
                       help='Skip dependency installation')
    
    args = parser.parse_args()
    
    setup = AISetupManager()
    
    print("🚀 AI Pipeline Setup Script")
    print("="*50)
    
    if args.verify_only:
        # Only verify existing setup
        logger.info("🔍 Verification mode - checking existing setup...")
        model_status = setup.verify_all_models()
        all_ready = all(model_status.values())
        
        if all_ready:
            logger.info("✅ All models verified successfully!")
            setup.test_complete_pipeline()
        else:
            failed_models = [name for name, status in model_status.items() if not status]
            logger.error("❌ Some models failed: %s", failed_models)
            return 1
            
    elif args.models_only:
        # Only set up models
        if not setup.setup_model_directories():
            return 1
        if not setup.setup_whisper_model():
            return 1
        setup.verify_all_models()
        
    elif args.quick_start:
        # Complete setup process
        logger.info("🚀 Starting complete setup process...")
        
        # Check system requirements
        if not setup.check_system_requirements():
            logger.error("❌ System requirements not met")
            return 1
        
        # Install dependencies (unless skipped)
        if not args.skip_deps:
            if not setup.install_dependencies():
                logger.error("❌ Failed to install dependencies")
                return 1
        
        # Set up models
        if not setup.setup_model_directories():
            return 1
            
        if not setup.setup_whisper_model():
            return 1
        
        # Create .env file
        setup.create_env_file()
        
        # Verify all models
        model_status = setup.verify_all_models()
        all_ready = all(model_status.values())
        
        if not all_ready:
            failed_models = [name for name, status in model_status.items() if not status]
            logger.warning("⚠️ Some models not ready: %s", failed_models)
            logger.info("ℹ️ You may need to download/configure these models manually")
        
        # Test pipeline
        setup.test_complete_pipeline()
        
        # Print next steps
        setup.print_next_steps()
        
    else:
        parser.print_help()
        return 0
    
    return 0

if __name__ == "__main__":
    exit(main())
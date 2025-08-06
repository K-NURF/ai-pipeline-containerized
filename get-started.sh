#!/bin/bash

# AI Pipeline - Quick Start Script
# This script helps new users get up and running quickly

set -e  # Exit on error

echo "🚀 AI Pipeline - Quick Start Script"
echo "=================================="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️ $1${NC}"
}

# Check if Python 3.11+ is available
check_python() {
    if command -v python3.11 &> /dev/null; then
        PYTHON_CMD="python3.11"
    elif command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
        if (( $(echo "$PYTHON_VERSION >= 3.11" | bc -l) )); then
            PYTHON_CMD="python3"
        else
            print_error "Python 3.11+ is required. Current version: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3.11+ is not installed."
        print_info "Install Python 3.11+ and try again."
        exit 1
    fi
    
    print_status "Found Python: $($PYTHON_CMD --version)"
}

# Check if we're in the right directory
check_directory() {
    if [[ ! -f "setup.py" ]] || [[ ! -f "requirements.txt" ]]; then
        print_error "This script must be run from the ai-pipeline-containerized directory"
        print_info "Make sure you've cloned the repository and are in the project root"
        exit 1
    fi
    
    print_status "Found project files"
}

# Check system requirements
check_system() {
    print_info "Checking system requirements..."
    
    # Check available memory
    if command -v free &> /dev/null; then
        MEMORY_GB=$(free -g | awk 'NR==2{printf "%.1f", $2}')
        if (( $(echo "$MEMORY_GB < 8" | bc -l) )); then
            print_warning "Low system memory: ${MEMORY_GB}GB (recommend 16GB+)"
        else
            print_status "System memory: ${MEMORY_GB}GB"
        fi
    fi
    
    # Check for GPU
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
            print_status "GPU detected: $GPU_INFO"
        else
            print_warning "nvidia-smi found but GPU not accessible"
        fi
    else
        print_info "No NVIDIA GPU detected (will use CPU)"
    fi
    
    # Check Docker
    if command -v docker &> /dev/null; then
        if docker --version &> /dev/null; then
            DOCKER_VERSION=$(docker --version | grep -oP '\d+\.\d+\.\d+')
            print_status "Docker found: $DOCKER_VERSION"
            DOCKER_AVAILABLE=true
        else
            print_warning "Docker installed but not accessible"
            DOCKER_AVAILABLE=false
        fi
    else
        print_info "Docker not found (manual setup will be used)"
        DOCKER_AVAILABLE=false
    fi
}

# Offer setup options
choose_setup_method() {
    echo
    echo "Choose your setup method:"
    echo "1) 🚀 Automated Python Setup (Recommended for development)"
    echo "2) 🐳 Docker Compose Setup (Recommended for production)"
    echo "3) 📋 Manual Setup Instructions"
    echo "4) ❌ Exit"
    echo
    
    read -p "Enter your choice (1-4): " choice
    
    case $choice in
        1)
            setup_python
            ;;
        2)
            if [[ "$DOCKER_AVAILABLE" == "true" ]]; then
                setup_docker
            else
                print_error "Docker is not available. Please install Docker first."
                print_info "Visit: https://docs.docker.com/get-docker/"
                exit 1
            fi
            ;;
        3)
            show_manual_instructions
            ;;
        4)
            echo "Goodbye!"
            exit 0
            ;;
        *)
            print_error "Invalid choice. Please run the script again."
            exit 1
            ;;
    esac
}

# Python setup method
setup_python() {
    print_info "Starting automated Python setup..."
    
    # Run the Python setup script
    print_info "Running setup script (this may take 10-15 minutes)..."
    if $PYTHON_CMD setup.py --quick-start; then
        print_status "Setup completed successfully!"
        show_next_steps_python
    else
        print_error "Setup failed. Check the output above for details."
        exit 1
    fi
}

# Docker setup method  
setup_docker() {
    print_info "Starting Docker Compose setup..."
    
    # Create .env file if it doesn't exist
    if [[ ! -f ".env" ]]; then
        if [[ -f ".env.example" ]]; then
            cp .env.example .env
            print_status "Created .env file from template"
        else
            print_warning ".env.example not found, using defaults"
        fi
    fi
    
    # Start Docker Compose
    print_info "Starting services with Docker Compose..."
    if docker-compose up -d; then
        print_status "Services started successfully!"
        
        # Wait for services to be ready
        print_info "Waiting for services to be ready..."
        sleep 10
        
        # Test health endpoint
        if curl -f http://localhost:8123/health &> /dev/null; then
            print_status "Services are healthy!"
            show_next_steps_docker
        else
            print_warning "Services started but health check failed"
            print_info "Check logs with: docker-compose logs"
        fi
    else
        print_error "Failed to start Docker services"
        exit 1
    fi
}

# Show manual instructions
show_manual_instructions() {
    echo
    echo "📋 Manual Setup Instructions"
    echo "============================"
    echo
    echo "1. Install dependencies:"
    echo "   pip install -r requirements.txt"
    echo
    echo "2. Download spaCy model:"
    echo "   python -m spacy download en_core_web_md"
    echo
    echo "3. Set up models:"
    echo "   python -c \"from app.models.whisper_model import WhisperModel; WhisperModel().load()\""
    echo
    echo "4. Create configuration:"
    echo "   cp .env.example .env"
    echo
    echo "5. Start Redis:"
    echo "   redis-server &"
    echo
    echo "6. Start Celery worker:"
    echo "   celery -A app.celery_app worker --loglevel=info -E &"
    echo
    echo "7. Start API server:"
    echo "   python -m app.main"
    echo
    echo "For detailed instructions, see: INSTALLATION.md"
}

# Show next steps for Python setup
show_next_steps_python() {
    echo
    echo "🎉 Setup Complete!"
    echo "================="
    echo
    echo "Next steps:"
    echo
    print_info "1. Start Redis server:"
    echo "   redis-server &"
    echo
    print_info "2. Start Celery worker:"
    echo "   celery -A app.celery_app worker --loglevel=info -E &"
    echo
    print_info "3. Start the API server:"
    echo "   python -m app.main"
    echo
    print_info "4. Test the API:"
    echo "   curl http://localhost:8123/health/detailed"
    echo
    print_info "5. View API documentation:"
    echo "   Open: http://localhost:8123/docs"
    echo
    echo "📚 For more information, see:"
    echo "   - README.md"
    echo "   - INSTALLATION.md"
    echo "   - docs/model-setup-guide.md"
}

# Show next steps for Docker setup
show_next_steps_docker() {
    echo
    echo "🎉 Docker Setup Complete!"
    echo "========================="
    echo
    echo "Your services are running!"
    echo
    print_info "API Server: http://localhost:8123"
    print_info "API Docs: http://localhost:8123/docs"
    print_info "Health Check: http://localhost:8123/health/detailed"
    echo
    print_info "Useful Docker commands:"
    echo "   docker-compose logs -f          # View logs"
    echo "   docker-compose ps               # Check status"
    echo "   docker-compose restart          # Restart services"
    echo "   docker-compose down             # Stop services"
    echo
    print_info "Test with an audio file:"
    echo "   curl -X POST -F 'audio=@sample.wav' -F 'language=sw' \\"
    echo "        http://localhost:8123/audio/process"
}

# Main execution
main() {
    echo "Starting AI Pipeline setup process..."
    echo
    
    check_directory
    check_python
    check_system
    choose_setup_method
}

# Run main function
main
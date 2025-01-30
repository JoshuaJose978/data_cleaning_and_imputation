#!/bin/bash

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print step headers
print_step() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

# Function to check if a command was successful
check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $1 successful${NC}"
    else
        echo -e "${RED}✗ Error: $1 failed${NC}"
        exit 1
    fi
}

# Function to check if Python is installed
check_python() {
    if ! command -v python &> /dev/null; then
        echo -e "${RED}Error: Python is not installed${NC}"
        echo "Please install Python and try again"
        exit 1
    fi
}

# Function to check if pip is installed
check_pip() {
    if ! command -v pip &> /dev/null; then
        echo -e "${RED}Error: pip is not installed${NC}"
        echo "Please install pip and try again"
        exit 1
    fi
}

# Function to create virtual environment
create_venv() {
    print_step "Creating Python virtual environment"
    python -m venv venv
    check_status "Virtual environment creation"
    
    # Activate virtual environment
    source venv/Scripts/activate
    check_status "Virtual environment activation"
}

# Function to install Python requirements
install_requirements() {
    print_step "Installing Python requirements"
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        check_status "Requirements installation"
    else
        echo -e "${RED}Error: requirements.txt not found${NC}"
        exit 1
    fi
}

# Function to run the Python script
run_python_script() {
    print_step "Running data generation and imputation pipeline"
    python main.py
    check_status "Python script execution"
}

# Function to check if data folder exists and contains required files
check_data_folder() {
    print_step "Checking data folder"
    if [ ! -d "data" ]; then
        mkdir data
        check_status "Data directory creation"
    fi
    
    if [ ! -f "data/imputation_metrics.csv" ]; then
        echo -e "${RED}Error: imputation_metrics.csv not found in data folder${NC}"
        exit 1
    fi
}

# Function to start the HTTP server
start_server() {
    print_step "Starting HTTP server"
    echo -e "${GREEN}Server starting at http://localhost:8000${NC}"
    echo -e "${BLUE}Press Ctrl+C to stop the server${NC}"
    python -m http.server 8000
}

# Function to open the browser (platform-dependent)
open_browser() {
    print_step "Opening dashboard in browser"
    # Try to open browser based on OS
    case "$(uname -s)" in
        Darwin*)    # macOS
            open http://localhost:8000
            ;;
        Linux*)     # Linux
            if command -v xdg-open &> /dev/null; then
                xdg-open http://localhost:8000
            elif command -v gnome-open &> /dev/null; then
                gnome-open http://localhost:8000
            fi
            ;;
        CYGWIN*|MINGW32*|MSYS*|MINGW*)  # Windows
            start http://localhost:8000
            ;;
        *)
            echo -e "${BLUE}Please open http://localhost:8000 in your browser${NC}"
            ;;
    esac
}

# Main execution
main() {
    print_step "Starting Imputation Dashboard Setup"
    
    # Check prerequisites
    check_python
    check_pip
    
    # Create and activate virtual environment
    create_venv
    
    # Install requirements
    install_requirements
    
    # Run Python script
    run_python_script
    
    # Check data folder and files
    check_data_folder
    
    # Open browser in background
    open_browser &
    
    # Start server
    start_server
}

# Run main function
main
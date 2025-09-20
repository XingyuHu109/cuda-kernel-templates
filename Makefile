# Simple Makefile for CUDA kernel templates

# CUDA compiler
NVCC = nvcc

# Compiler flags
NVCCFLAGS = -O2 -arch=sm_50

# Target executable
TARGET = vector_add_template

# Source file
SOURCE = vector_add_template.cu

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) $(SOURCE) -o $(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Help target
help:
	@echo "Available targets:"
	@echo "  all     - Build the template (default)"
	@echo "  clean   - Remove build artifacts"
	@echo "  run     - Build and run the template"
	@echo "  help    - Show this help message"

.PHONY: all clean run help
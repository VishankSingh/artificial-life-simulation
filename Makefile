TARGET = main

NVCC = nvcc

SRC = main.cu

LIBS = -lGL -lGLU -lglfw

NVCC_FLAGS =

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LIBS)


clean:
	rm -f $(TARGET)

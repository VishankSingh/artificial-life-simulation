TARGET = simulation

NVCC = nvcc

SRC = main.cu

LIBS = -lGL -lGLU -lglfw

CPPFLAGS = -Iinclude
NVCC_FLAGS =
LDFLAGS = $(LIBS)

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)

run:
	./$(TARGET)

.PHONY: all clean

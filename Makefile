CXX := g++
CXXFLAGS := -g -Wall -O2 -mavx512f -march=native
LDFLAGS := -fopenmp -lopenblas -lpthread -lgfortran \
		   -LOpenBLAS/build/lib \
		   -Lgoogletest/build/lib
INCLUDES := -IOpenBLAS/build/include \
			-Igoogletest/googlemock/include \
			-Igoogletest/googletest/include

SRC_DIR := src
BUILD_DIR := build
SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRC_FILES))
TARGET := main.out

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) -c $< -o $@

$(TARGET): $(OBJ_FILES)
	$(CXX) -o $@ $^ $(LDFLAGS) 

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: clean

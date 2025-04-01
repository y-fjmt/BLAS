CXX := g++
CXXFLAGS := -g -O2 -mavx512f
LDFLAGS := -L/home/fujimoto/opt/OpenBLAS/lib -fopenmp -lopenblas -lpthread -lgfortran
INCLUDES := -I/home/fujimoto/opt/OpenBLAS/include/

SRC_DIR := src
BUILD_DIR := build
SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRC_FILES))
TARGET := main.out

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# compile
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) -c $< -o $@

# link
$(TARGET): $(OBJ_FILES)
	$(CXX) -o $@ $^ $(LDFLAGS)

# clean up
clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: clean
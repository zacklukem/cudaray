TARGET_EXEC := cudaray

BUILD_DIR := ./build
SRC_DIRS := ./src

SRCS := $(shell find $(SRC_DIRS) -name *.cu)

OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)

INC_DIRS := $(shell find $(SRC_DIRS) -type d) lib/glm lib/stb lib/thrust lib/csl usr/local/boost
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CXX = /opt/cuda-11.2/bin/nvcc

# Link
$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
	echo "[" $(TARGET_EXEC) "] Linking..."
	$(CXX) $(OBJS) -o $@ $(LDFLAGS) -Xptxas='-suppress-stack-size-warning'

# Build
$(BUILD_DIR)/%.cu.o: %.cu
	echo "[" $< "] Compiling..."
	mkdir -p $(dir $@)
	$(CXX) -w -dc $(INC_FLAGS) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@ -Xptxas='-suppress-stack-size-warning'

.PHONY: clean
clean:
	rm -r $(BUILD_DIR)

-include $(DEPS)
TARGET_EXEC := cudaray

BUILD_DIR := ./build
SRC_DIRS := ./src

SRCS := $(shell find $(SRC_DIRS) -name *.cu)

OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)

INC_DIRS := $(shell find $(SRC_DIRS) -type d) lib/glm lib/stb lib/thrust lib/csl
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CXX = /opt/cuda-11.2/bin/nvcc

# Link
$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

# Build
$(BUILD_DIR)/%.cu.o: %.cu
	mkdir -p $(dir $@)
	$(CXX) -w -dc $(INC_FLAGS) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@


.PHONY: clean
clean:
	rm -r $(BUILD_DIR)

-include $(DEPS)
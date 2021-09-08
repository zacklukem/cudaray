#ifndef SRC_OBJ
#define SRC_OBJ

#include <boost/algorithm/string.hpp>
#include <fstream>
#include <glm.inl>
#include <string>
#include <vector>

namespace cudaray {

struct ObjModel {
  std::vector<glm::vec3> vertices;
  std::vector<glm::ivec3> faces;
};

ObjModel loadObj(const char* path) {
  ObjModel model;

  std::string line;
  std::ifstream file(path);
  if (file.is_open()) {
    while (std::getline(file, line)) {
      if (line.length() < 1)
        continue;
      if (line.at(0) == 'v') {
        std::vector<std::string> split;
        boost::split(split, line, boost::is_any_of(" "));
        model.vertices.push_back(glm::vec3(std::stof(split.at(1)),
                                           std::stof(split.at(2)),
                                           std::stof(split.at(3))));
      } else if (line.at(0) == 'f') {
        std::vector<std::string> split;
        boost::split(split, line, boost::is_any_of(" "));
        model.faces.push_back(glm::ivec3(std::stoi(split.at(1)),
                                         std::stoi(split.at(2)),
                                         std::stoi(split.at(3))));
      }
    }
    file.close();
  }

  return model;
}

} // namespace cudaray

#endif // SRC_OBJ

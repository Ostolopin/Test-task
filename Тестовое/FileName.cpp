#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <limits>
#include <string>

// OpenGL и связанные библиотеки
#include <GL/glew.h>
#include <GLFW/glfw3.h>

// GLM для работы с матрицами и векторами
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Eigen для выполнения PCA
#include <Eigen/Dense>

// Для использования M_PI
#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Logging setup
enum LogLevel {
    DEBUG,
    INFO,
    ERROR
};

LogLevel currentLogLevel = DEBUG;

std::ofstream logFile;

void log(LogLevel level, const std::string& message) {
    if (level >= currentLogLevel) {
        switch (level) {
        case DEBUG:
            logFile << "[DEBUG]: " << message << std::endl;
            break;
        case INFO:
            logFile << "[INFO]: " << message << std::endl;
            break;
        case ERROR:
            logFile << "[ERROR]: " << message << std::endl;
            break;
        }
    }
}

// Data structures
struct Section {
    int Z = 0;
    std::vector<std::pair<int, int>> points;
};

struct Header {
    uint16_t version;
    int N;
    uint16_t pointsPerSection;
    uint16_t logIndex;
    double datetime;
    uint16_t frontDiameter;
    uint16_t middleDiameter;
    uint16_t backDiameter;
    uint16_t tipDiameter;
    uint16_t logLength;
    uint8_t curvature;
    int16_t curvatureDirection;
    int16_t taper;
    int16_t taperBase;
    float physicalVolume;
    uint16_t flags;
    float encoderPulsePrice;
};

// Global variables for control
float rotationX = 0.0f;
float rotationY = 0.0f;
float scale = 1.0f;
float positionX = 0.0f;

Header header;

std::vector<Section> sections;
std::vector<Section> cylinderSections; // Sections for the cylinder
GLuint VAO, VBO;
GLuint cylinderVAO, cylinderVBO;

double lastX = 400, lastY = 300;
bool firstMouse = true;
bool mousePressed = false;
float sensitivity = 0.005f;

bool cylinderCreated = false;

// Определяем numSegments глобально
const int numSegments = 36; // Number of points around the circle

// Function prototypes
Header readHeader(std::ifstream& file);
std::vector<Section> readSections(std::ifstream& file, int N, std::streampos fileSize);
void processInput(GLFWwindow* window);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
GLuint compileShader(GLenum type, const char* source);
size_t createBuffersFromSections();
void createCylinderModel(const Header& originalHeader);
void computeSectionCenters();
void computeModelAxis(Eigen::Vector3f& axisDirection);
void alignModelToAxis(const Eigen::Vector3f& axisDirection);
float computeMinRadius();
void createCylinderSections(float minRadius);
void saveCylinderModel(const std::string& filename, const Header& originalHeader);

// Shader sources
const char* vertexShaderSource = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
uniform vec4 ourColor;
out vec4 FragColor;
void main() {
    FragColor = ourColor;
}
)";

// Data structures for centered sections
struct CenteredSection {
    int Z = 0;
    float centerX = 0.0f;
    float centerY = 0.0f;
    std::vector<std::pair<float, float>> centeredPoints;
};

std::vector<CenteredSection> centeredSections;

// Function to read header
Header readHeader(std::ifstream& file) {
    Header header;
    file.read(reinterpret_cast<char*>(&header.version), sizeof(header.version));
    file.read(reinterpret_cast<char*>(&header.N), sizeof(header.N));
    file.read(reinterpret_cast<char*>(&header.pointsPerSection), sizeof(header.pointsPerSection));
    file.read(reinterpret_cast<char*>(&header.logIndex), sizeof(header.logIndex));
    file.read(reinterpret_cast<char*>(&header.datetime), sizeof(header.datetime));
    file.read(reinterpret_cast<char*>(&header.frontDiameter), sizeof(header.frontDiameter));
    file.read(reinterpret_cast<char*>(&header.middleDiameter), sizeof(header.middleDiameter));
    file.read(reinterpret_cast<char*>(&header.backDiameter), sizeof(header.backDiameter));
    file.read(reinterpret_cast<char*>(&header.tipDiameter), sizeof(header.tipDiameter));
    file.read(reinterpret_cast<char*>(&header.logLength), sizeof(header.logLength));
    file.read(reinterpret_cast<char*>(&header.curvature), sizeof(header.curvature));
    file.seekg(1, std::ios::cur);
    file.read(reinterpret_cast<char*>(&header.curvatureDirection), sizeof(header.curvatureDirection));
    file.read(reinterpret_cast<char*>(&header.taper), sizeof(header.taper));
    file.read(reinterpret_cast<char*>(&header.taperBase), sizeof(header.taperBase));
    file.read(reinterpret_cast<char*>(&header.physicalVolume), sizeof(header.physicalVolume));
    file.read(reinterpret_cast<char*>(&header.flags), sizeof(header.flags));
    file.read(reinterpret_cast<char*>(&header.encoderPulsePrice), sizeof(header.encoderPulsePrice));
    file.seekg(83, std::ios::cur);
    return header;
}

// Function to read sections
std::vector<Section> readSections(std::ifstream& file, int N, std::streampos fileSize) {
    std::vector<Section> sections;
    for (int i = 0; i < N; ++i) {
        Section section;
        file.read(reinterpret_cast<char*>(&section.Z), sizeof(section.Z));
        log(INFO, "После чтения Z для сечения " + std::to_string(i) + ": позиция в файле = " + std::to_string(file.tellg()));
        log(INFO, "Сечение " + std::to_string(i) + ": Z = " + std::to_string(section.Z));
        uint16_t M;
        file.read(reinterpret_cast<char*>(&M), sizeof(M));
        log(INFO, "После чтения количества точек для сечения " + std::to_string(i) + ": позиция в файле = " + std::to_string(file.tellg()));
        log(INFO, "Количество точек в сечении " + std::to_string(i) + ": " + std::to_string(M));
        for (int j = 0; j < M; ++j) {
            int16_t x, y;
            file.read(reinterpret_cast<char*>(&x), sizeof(x));
            file.read(reinterpret_cast<char*>(&y), sizeof(y));
            section.points.emplace_back(x, y);
            log(INFO, "Точка " + std::to_string(j) + ": X = " + std::to_string(x) + ", Y = " + std::to_string(y));
            log(INFO, "После чтения точки " + std::to_string(j) + " для сечения " + std::to_string(i) + ": позиция в файле = " + std::to_string(file.tellg()));
        }
        sections.push_back(section);
        if (file.tellg() >= fileSize) {
            log(ERROR, "Файл прочитан до конца на сечении " + std::to_string(i));
            break;
        }
    }
    return sections;
}

// Input processing
void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        positionX -= 1.0f;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        positionX += 1.0f;
    // Check for 'C' key to create cylinder
    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS && !cylinderCreated) {
        createCylinderModel(header); // Передаем оригинальный заголовок
        cylinderCreated = true;
    }
}

// Mouse button callback
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            mousePressed = true;
            firstMouse = true;
        }
        else if (action == GLFW_RELEASE) {
            mousePressed = false;
        }
    }
}

// Mouse movement callback
void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (mousePressed) {
        if (firstMouse) {
            lastX = xpos;
            lastY = ypos;
            firstMouse = false;
        }
        float xoffset = static_cast<float>(xpos - lastX);
        float yoffset = static_cast<float>(ypos - lastY);
        lastX = xpos;
        lastY = ypos;
        rotationY += xoffset * sensitivity;
        rotationX += yoffset * sensitivity;
    }
}

// Scroll callback for zooming
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    scale += static_cast<float>(yoffset) * 0.1f;
    if (scale < 0.1f)
        scale = 0.1f;
}

// Shader compilation
GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    return shader;
}

// Проверка, находится ли точка внутри полигона
bool isPointInsidePolygon(const std::vector<std::pair<float, float>>& polygon, float x, float y) {
    int intersections = 0;
    size_t count = polygon.size();
    for (size_t i = 0; i < count; ++i) {
        auto p1 = polygon[i];
        auto p2 = polygon[(i + 1) % count];
        if (((p1.second > y) != (p2.second > y)) &&
            (x < (p2.first - p1.first) * (y - p1.second) / (p2.second - p1.second + 1e-6f) + p1.first)) {
            intersections++;
        }
    }
    return (intersections % 2) == 1;
}

// Генерация точек на поверхности цилиндра
std::vector<Eigen::Vector3f> sampleCylinderSurface(const Eigen::Vector3f& baseCenter, const Eigen::Vector3f& axisDirection, float height, float radius, int numHeightSamples, int numAngleSamples) {
    std::vector<Eigen::Vector3f> points;
    for (int i = 0; i <= numHeightSamples; ++i) {
        float t = static_cast<float>(i) / numHeightSamples; // от 0 до 1
        float z = t * height;
        for (int j = 0; j < numAngleSamples; ++j) {
            float theta = 2.0f * M_PI * j / numAngleSamples;
            float x = radius * std::cos(theta);
            float y = radius * std::sin(theta);
            // Позиция точки на цилиндре
            Eigen::Vector3f point = baseCenter + axisDirection * z + Eigen::Vector3f(x, y, 0.0f);
            points.push_back(point);
        }
    }
    return points;
}

// Проверка, находится ли точка внутри сечения модели
bool isPointInsideModel(const Eigen::Vector3f& point, const std::vector<Section>& modelSections) {
    // Находим ближайшее сечение по Z
    int closestIndex = -1;
    float minZDiff = std::numeric_limits<float>::max();
    for (size_t i = 0; i < modelSections.size(); ++i) {
        float z = static_cast<float>(modelSections[i].Z) / 10.0f;
        float zDiff = std::abs(point.z() - z);
        if (zDiff < minZDiff) {
            minZDiff = zDiff;
            closestIndex = static_cast<int>(i);
        }
    }
    if (closestIndex == -1) {
        return false;
    }

    // Проверяем, находится ли точка внутри полигона сечения
    const auto& section = modelSections[closestIndex];
    std::vector<std::pair<float, float>> polygon;
    for (const auto& p : section.points) {
        float x = static_cast<float>(p.first) / 10.0f;
        float y = static_cast<float>(p.second) / 10.0f;
        polygon.emplace_back(x, y);
    }

    return isPointInsidePolygon(polygon, point.x(), point.y());
}



// Вычисление минимального расстояния от точки до границы полигона
float minDistanceToPolygonEdge(const std::vector<std::pair<float, float>>& polygon, float x, float y) {
    float minDist = std::numeric_limits<float>::max();
    size_t count = polygon.size();
    for (size_t i = 0; i < count; ++i) {
        auto p1 = polygon[i];
        auto p2 = polygon[(i + 1) % count];
        // Вычисляем расстояние от точки до отрезка (p1, p2)
        float dx = p2.first - p1.first;
        float dy = p2.second - p1.second;
        float lengthSquared = dx * dx + dy * dy;
        float t = ((x - p1.first) * dx + (y - p1.second) * dy) / lengthSquared;
        t = std::max(0.0f, std::min(1.0f, t));
        float projX = p1.first + t * dx;
        float projY = p1.second + t * dy;
        float dist = std::sqrt((x - projX) * (x - projX) + (y - projY) * (y - projY));
        if (dist < minDist) {
            minDist = dist;
        }
    }
    return minDist;
}

void findMaxInscribedCircle(const CenteredSection& section, float& centerX, float& centerY, float& radius) {
    // Определяем границы сечения
    float minX = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float minY = std::numeric_limits<float>::max();
    float maxY = std::numeric_limits<float>::lowest();
    for (const auto& p : section.centeredPoints) {
        if (p.first < minX) minX = p.first;
        if (p.first > maxX) maxX = p.first;
        if (p.second < minY) minY = p.second;
        if (p.second > maxY) maxY = p.second;
    }
    // Создаем сетку точек внутри границ
    int gridSize = 150; // Чем больше значение, тем точнее, но медленнее
    float stepX = (maxX - minX) / gridSize;
    float stepY = (maxY - minY) / gridSize;
    float maxRadius = 0.0f;
    float bestX = section.centerX;
    float bestY = section.centerY;
    std::vector<std::pair<float, float>> polygon = section.centeredPoints;
    for (int i = 0; i <= gridSize; ++i) {
        for (int j = 0; j <= gridSize; ++j) {
            float x = minX + i * stepX;
            float y = minY + j * stepY;
            if (isPointInsidePolygon(polygon, x, y)) {
                float dist = minDistanceToPolygonEdge(polygon, x, y);
                if (dist > maxRadius) {
                    maxRadius = dist;
                    bestX = x;
                    bestY = y;
                }
            }
        }
    }
    centerX = bestX + section.centerX; // Возвращаем к исходным координатам
    centerY = bestY + section.centerY;
    radius = maxRadius;
}


// Функция для нахождения лучшей прямой через центры сечений
void computeBestFitLine(Eigen::Vector3f& pointOnLine, Eigen::Vector3f& lineDirection) {
    size_t N = centeredSections.size();
    Eigen::MatrixXf positions(N, 3);
    for (size_t i = 0; i < N; ++i) {
        positions(i, 0) = centeredSections[i].centerX;
        positions(i, 1) = centeredSections[i].centerY;
        positions(i, 2) = static_cast<float>(centeredSections[i].Z) / 10.0f;
    }
    // Вычисляем среднее положение
    pointOnLine = positions.colwise().mean();
    // Вычитаем среднее
    positions.rowwise() -= pointOnLine.transpose();
    // Выполняем SVD
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(positions, Eigen::ComputeThinU | Eigen::ComputeThinV);
    // Направление линии соответствует первому сингулярному вектору
    lineDirection = svd.matrixV().col(0);
}


// Create buffers from sections and add end caps
size_t createBuffersFromSections() {
    std::vector<float> vertices;

    // Generate side triangles
    for (size_t i = 0; i < sections.size() - 1; ++i) {
        const auto& section1 = sections[i];
        const auto& section2 = sections[i + 1];
        size_t M1 = section1.points.size();
        size_t M2 = section2.points.size();
        size_t M = std::min(M1, M2);
        if (M < 3) continue;
        for (size_t j = 0; j < M; ++j) {
            size_t nextJ = (j + 1) % M;
            float x0 = static_cast<float>(section1.points[j].first) / 10.0f;
            float y0 = static_cast<float>(section1.points[j].second) / 10.0f;
            float z0 = static_cast<float>(section1.Z) / 10.0f;
            float x1 = static_cast<float>(section1.points[nextJ].first) / 10.0f;
            float y1 = static_cast<float>(section1.points[nextJ].second) / 10.0f;
            float z1 = static_cast<float>(section1.Z) / 10.0f;
            float x2 = static_cast<float>(section2.points[j].first) / 10.0f;
            float y2 = static_cast<float>(section2.points[j].second) / 10.0f;
            float z2 = static_cast<float>(section2.Z) / 10.0f;
            float x3 = static_cast<float>(section2.points[nextJ].first) / 10.0f;
            float y3 = static_cast<float>(section2.points[nextJ].second) / 10.0f;
            float z3 = static_cast<float>(section2.Z) / 10.0f;
            // First triangle
            vertices.push_back(x0); vertices.push_back(y0); vertices.push_back(z0);
            vertices.push_back(x1); vertices.push_back(y1); vertices.push_back(z1);
            vertices.push_back(x2); vertices.push_back(y2); vertices.push_back(z2);
            // Second triangle
            vertices.push_back(x1); vertices.push_back(y1); vertices.push_back(z1);
            vertices.push_back(x3); vertices.push_back(y3); vertices.push_back(z3);
            vertices.push_back(x2); vertices.push_back(y2); vertices.push_back(z2);
        }
    }

    // Add end cap for the first section
    {
        const auto& section = sections.front();
        size_t M = section.points.size();
        if (M >= 3) {
            float z0 = static_cast<float>(section.Z) / 10.0f;
            float centerX = 0.0f;
            float centerY = 0.0f;
            for (const auto& p : section.points) {
                centerX += static_cast<float>(p.first) / 10.0f;
                centerY += static_cast<float>(p.second) / 10.0f;
            }
            centerX /= M;
            centerY /= M;
            for (size_t j = 0; j < M; ++j) {
                size_t nextJ = (j + 1) % M;
                float x1 = static_cast<float>(section.points[j].first) / 10.0f;
                float y1 = static_cast<float>(section.points[j].second) / 10.0f;
                float x2 = static_cast<float>(section.points[nextJ].first) / 10.0f;
                float y2 = static_cast<float>(section.points[nextJ].second) / 10.0f;
                // Triangle
                vertices.push_back(centerX); vertices.push_back(centerY); vertices.push_back(z0);
                vertices.push_back(x1); vertices.push_back(y1); vertices.push_back(z0);
                vertices.push_back(x2); vertices.push_back(y2); vertices.push_back(z0);
            }
        }
    }

    // Add end cap for the last section
    {
        const auto& section = sections.back();
        size_t M = section.points.size();
        if (M >= 3) {
            float z0 = static_cast<float>(section.Z) / 10.0f;
            float centerX = 0.0f;
            float centerY = 0.0f;
            for (const auto& p : section.points) {
                centerX += static_cast<float>(p.first) / 10.0f;
                centerY += static_cast<float>(p.second) / 10.0f;
            }
            centerX /= M;
            centerY /= M;
            for (size_t j = 0; j < M; ++j) {
                size_t nextJ = (j + 1) % M;
                float x1 = static_cast<float>(section.points[j].first) / 10.0f;
                float y1 = static_cast<float>(section.points[j].second) / 10.0f;
                float x2 = static_cast<float>(section.points[nextJ].first) / 10.0f;
                float y2 = static_cast<float>(section.points[nextJ].second) / 10.0f;
                // Triangle
                vertices.push_back(centerX); vertices.push_back(centerY); vertices.push_back(z0);
                vertices.push_back(x2); vertices.push_back(y2); vertices.push_back(z0);
                vertices.push_back(x1); vertices.push_back(y1); vertices.push_back(z0);
            }
        }
    }

    // Upload data to buffers
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    log(INFO, "Количество вершин для отрисовки: " + std::to_string(vertices.size()));
    return vertices.size();
}

// Function to compute centers of sections
void computeSectionCenters() {
    centeredSections.clear();
    for (const auto& section : sections) {
        CenteredSection centeredSection;
        centeredSection.Z = section.Z;
        size_t M = section.points.size();
        float sumX = 0.0f;
        float sumY = 0.0f;
        for (const auto& p : section.points) {
            float x = static_cast<float>(p.first) / 10.0f;
            float y = static_cast<float>(p.second) / 10.0f;
            sumX += x;
            sumY += y;
        }
        centeredSection.centerX = sumX / M;
        centeredSection.centerY = sumY / M;
        for (const auto& p : section.points) {
            float x = static_cast<float>(p.first) / 10.0f - centeredSection.centerX;
            float y = static_cast<float>(p.second) / 10.0f - centeredSection.centerY;
            centeredSection.centeredPoints.emplace_back(x, y);
        }
        centeredSections.push_back(centeredSection);
    }
}

// Function to compute model axis using PCA
Eigen::Matrix3f computeModelAxis(Eigen::Vector3f& axisDirection, Eigen::Vector3f& meanPosition) {
    // Собираем центры сечений в матрицу
    Eigen::MatrixXf data(centeredSections.size(), 3);
    for (size_t i = 0; i < centeredSections.size(); ++i) {
        data(i, 0) = centeredSections[i].centerX;
        data(i, 1) = centeredSections[i].centerY;
        data(i, 2) = static_cast<float>(centeredSections[i].Z) / 10.0f;
    }
    // Вычитаем среднее
    meanPosition = data.colwise().mean();
    data.rowwise() -= meanPosition.transpose();
    // Вычисляем матрицу ковариации
    Eigen::Matrix3f cov = data.transpose() * data;
    // Вычисляем собственные значения и векторы
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(cov);
    axisDirection = eig.eigenvectors().col(2).normalized();

    // Определяем вектор поворота
    Eigen::Vector3f zAxis(0.0f, 0.0f, 1.0f);
    Eigen::Vector3f rotationAxis = axisDirection.cross(zAxis);
    float angle = std::acos(axisDirection.dot(zAxis));

    // Создаем матрицу поворота
    Eigen::Matrix3f rotationMatrix = Eigen::Matrix3f::Identity();
    if (rotationAxis.norm() > 1e-6) {
        rotationAxis.normalize();
        Eigen::AngleAxisf rotation(angle, rotationAxis);
        rotationMatrix = rotation.toRotationMatrix();
    }

    return rotationMatrix;
}


// Function to align model to principal axis
void alignModelToAxis(const Eigen::Matrix3f& rotationMatrix, const Eigen::Vector3f& meanPosition) {
    for (auto& section : centeredSections) {
        // Поворот центра сечения
        Eigen::Vector3f center(section.centerX, section.centerY, static_cast<float>(section.Z) / 10.0f);
        center -= meanPosition;
        center = rotationMatrix * center;
        section.centerX = center.x();
        section.centerY = center.y();
        section.Z = static_cast<int>((center.z()) * 10.0f);
        // Поворот точек сечения
        for (auto& point : section.centeredPoints) {
            Eigen::Vector3f p(point.first, point.second, 0.0f);
            p = rotationMatrix * p;
            point.first = p.x();
            point.second = p.y();
        }
    }
}

void applyInverseTransformationToCylinder(const Eigen::Matrix3f& rotationMatrix, const Eigen::Vector3f& meanPosition) {
    Eigen::Matrix3f inverseRotation = rotationMatrix.transpose(); // Обратная матрица поворота
    for (auto& section : cylinderSections) {
        // Корректируем Z
        float z = static_cast<float>(section.Z) / 10.0f;
        Eigen::Vector3f center(0.0f, 0.0f, z);
        center = inverseRotation * center + meanPosition;
        section.Z = static_cast<int>(center.z() * 10.0f);

        // Корректируем точки
        for (auto& point : section.points) {
            float x = static_cast<float>(point.first) / 10.0f;
            float y = static_cast<float>(point.second) / 10.0f;
            Eigen::Vector3f p(x, y, z);
            p = inverseRotation * p + meanPosition;
            point.first = static_cast<int16_t>(p.x() * 10.0f);
            point.second = static_cast<int16_t>(p.y() * 10.0f);
        }
    }
}


// Function to compute minimum radius
float computeMinRadius() {
    float minRadius = std::numeric_limits<float>::max();
    for (const auto& section : centeredSections) {
        for (const auto& point : section.centeredPoints) {
            float distance = std::sqrt(point.first * point.first + point.second * point.second);
            if (distance < minRadius) {
                minRadius = distance;
            }
        }
    }
    return minRadius;
}

// Function to create cylinder sections
void createCylinderSections(float minRadius) {
    cylinderSections.clear();
    for (const auto& section : centeredSections) {
        Section cylSection;
        cylSection.Z = section.Z;
        for (int i = 0; i < numSegments; ++i) {
            float angle = 2.0f * static_cast<float>(M_PI) * i / numSegments;
            float x = minRadius * std::cos(angle);
            float y = minRadius * std::sin(angle);
            // Смещаем к центру сечения
            int16_t xi = static_cast<int16_t>((x + section.centerX) * 10.0f);
            int16_t yi = static_cast<int16_t>((y + section.centerY) * 10.0f);
            cylSection.points.emplace_back(xi, yi);
        }
        cylinderSections.push_back(cylSection);
    }
}

// Function to save cylinder model
void saveCylinderModel(const std::string& filename, const Header& originalHeader) {
    std::ofstream outFile(filename, std::ios::binary);
    if (outFile.is_open()) {
        // Copy the original header and adjust necessary fields
        Header cylHeader = originalHeader;
        cylHeader.N = static_cast<int>(cylinderSections.size());
        // Записываем заголовок
        outFile.write(reinterpret_cast<const char*>(&cylHeader), sizeof(cylHeader));
        // Write sections
        for (const auto& section : cylinderSections) {
            outFile.write(reinterpret_cast<const char*>(&section.Z), sizeof(section.Z));
            uint16_t M = static_cast<uint16_t>(section.points.size());
            outFile.write(reinterpret_cast<const char*>(&M), sizeof(M));
            for (const auto& point : section.points) {
                int16_t x = point.first;
                int16_t y = point.second;
                outFile.write(reinterpret_cast<const char*>(&x), sizeof(x));
                outFile.write(reinterpret_cast<const char*>(&y), sizeof(y));
            }
        }
        outFile.close();
        log(INFO, "Цилиндр сохранен в файл " + filename);
    }
    else {
        log(ERROR, "Не удалось сохранить цилиндр в файл " + filename);
    }
}

// Function to create the maximum cylinder model
void createCylinderModel(const Header& originalHeader) {
    // Вычисляем центры сечений
    computeSectionCenters();

    // Нахождение максимальных вписанных кругов в каждом сечении
    std::vector<float> radii;
    std::vector<Eigen::Vector3f> circleCenters;
    for (const auto& section : centeredSections) {
        float centerX, centerY, radius;
        findMaxInscribedCircle(section, centerX, centerY, radius);
        radii.push_back(radius);
        circleCenters.emplace_back(centerX, centerY, static_cast<float>(section.Z) / 10.0f);
    }

    // Фильтрация аномальных значений радиусов
    std::vector<float> filteredRadii;
    float radiusSum = 0.0f;
    for (float r : radii) {
        if (r > 0.0f) { // Исключаем нулевые и отрицательные значения
            filteredRadii.push_back(r);
            radiusSum += r;
        }
    }

    if (filteredRadii.empty()) {
        log(ERROR, "После фильтрации не осталось значений радиусов.");
        return;
    }

    // Вычисляем среднее значение радиуса
    float averageRadius = radiusSum / filteredRadii.size();

    // Определяем максимальное допустимое отклонение (например, 20% от среднего)
    float maxDeviation = averageRadius * 0.2f;

    // Повторно фильтруем радиусы, исключая выбросы
    std::vector<float> finalRadii;
    for (float r : filteredRadii) {
        if (std::abs(r - averageRadius) <= maxDeviation) {
            finalRadii.push_back(r);
        }
    }

    if (finalRadii.empty()) {
        log(ERROR, "После удаления выбросов не осталось значений радиусов.");
        return;
    }

    // Находим минимальный радиус из окончательного списка радиусов
    float minRadius = *std::min_element(finalRadii.begin(), finalRadii.end());

    // Применяем коэффициент безопасности
    float safetyFactor = 0.98f; // Уменьшаем радиус на 2%
    minRadius *= safetyFactor;
    log(INFO, "Минимальный радиус после применения коэффициента безопасности: " + std::to_string(minRadius));

    // Фитируем линию через центры максимальных вписанных кругов
    Eigen::Vector3f pointOnLine, lineDirection;
    {
        size_t N = circleCenters.size();
        Eigen::MatrixXf positions(N, 3);
        for (size_t i = 0; i < N; ++i) {
            positions.row(i) = circleCenters[i];
        }
        // Вычисляем среднее положение
        pointOnLine = positions.colwise().mean();
        // Вычитаем среднее
        positions.rowwise() -= pointOnLine.transpose();
        // Выполняем SVD
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(positions, Eigen::ComputeThinU | Eigen::ComputeThinV);
        // Направление линии соответствует первому сингулярному вектору
        lineDirection = svd.matrixV().col(0);
    }

    // Нормализуем направление линии
    lineDirection.normalize();

    // Определяем высоту цилиндра
    float cylinderHeight = (circleCenters.back().z() - circleCenters.front().z());

    // Определяем центр основания цилиндра
    Eigen::Vector3f baseCenter = pointOnLine;

    // Копируем исходные значения радиуса и цилиндрических секций
    float originalRadius = minRadius;
    float adjustedRadius = minRadius;

    // Параметры сэмплирования
    int numHeightSamples = 20; // Количество сечений по высоте цилиндра
    int numAngleSamples = 36;  // Количество точек по окружности

    float allowedPercentage = 5.0f; // Допустимый процент точек вне модели

    // Цикл для регулировки радиуса цилиндра
    while (true) {
        // Генерируем точки на поверхности цилиндра
        std::vector<Eigen::Vector3f> cylinderPoints = sampleCylinderSurface(baseCenter, lineDirection, cylinderHeight, adjustedRadius, numHeightSamples, numAngleSamples);

        // Проверяем, сколько точек находится вне модели
        int outsideCount = 0;
        for (const auto& point : cylinderPoints) {
            if (!isPointInsideModel(point, sections)) {
                outsideCount++;
            }
        }

        float outsidePercentage = 100.0f * outsideCount / cylinderPoints.size();
        log(INFO, "Процент точек цилиндра вне модели: " + std::to_string(outsidePercentage) + "%");

        if (outsidePercentage <= allowedPercentage) {
            // Допустимый процент достигнут, выходим из цикла
            break;
        }
        else {
            // Уменьшаем радиус цилиндра на небольшой шаг
            adjustedRadius *= 0.99f; // Уменьшаем радиус на 1%
            log(INFO, "Уменьшаем радиус цилиндра до: " + std::to_string(adjustedRadius));

            // Проверяем, не стал ли радиус слишком маленьким
            if (adjustedRadius < 0.1f) {
                log(ERROR, "Радиус цилиндра стал слишком малым.");
                return;
            }
        }
    }

    minRadius = adjustedRadius;

    // Пересоздаем цилиндрические сечения с новым радиусом
    cylinderSections.clear();
    for (size_t idx = 0; idx < circleCenters.size(); ++idx) {
        const auto& center = circleCenters[idx];
        // Проецируем центр на линию оси
        Eigen::Vector3f toCenter = center - pointOnLine;
        float t = toCenter.dot(lineDirection); // Положение вдоль оси
        Eigen::Vector3f projectedCenter = pointOnLine + t * lineDirection;

        Section cylSection;
        cylSection.Z = static_cast<int>(projectedCenter.z() * 10.0f);
        for (int i = 0; i < numSegments; ++i) {
            float angle = 2.0f * static_cast<float>(M_PI) * i / numSegments;
            float x = minRadius * std::cos(angle);
            float y = minRadius * std::sin(angle);
            // Поворачиваем точку в пространство модели
            Eigen::Vector3f radialVec = Eigen::Vector3f(x, y, 0.0f);
            Eigen::Vector3f normal = lineDirection.cross(Eigen::Vector3f(0, 0, 1)).normalized();
            Eigen::AngleAxisf rotation(acos(lineDirection.dot(Eigen::Vector3f(0, 0, 1))), normal);
            Eigen::Vector3f rotatedVec = rotation * radialVec;

            // Смещаем к центру сечения на оси
            Eigen::Vector3f point = projectedCenter + rotatedVec;
            int16_t xi = static_cast<int16_t>(point.x() * 10.0f);
            int16_t yi = static_cast<int16_t>(point.y() * 10.0f);
            cylSection.points.emplace_back(xi, yi);
        }
        cylinderSections.push_back(cylSection);
    }

    // Сохраняем цилиндр в файл
    saveCylinderModel("cylinder_model.lprf", originalHeader);

    // Генерируем OpenGL буферы для цилиндра
    // Создаем вершины для OpenGL
    std::vector<float> vertices;

    // Generate side triangles
    for (size_t i = 0; i < cylinderSections.size() - 1; ++i) {
        const auto& section1 = cylinderSections[i];
        const auto& section2 = cylinderSections[i + 1];
        size_t M = section1.points.size();
        for (size_t j = 0; j < M; ++j) {
            size_t nextJ = (j + 1) % M;
            float x0 = static_cast<float>(section1.points[j].first) / 10.0f;
            float y0 = static_cast<float>(section1.points[j].second) / 10.0f;
            float z0 = static_cast<float>(section1.Z) / 10.0f;
            float x1 = static_cast<float>(section1.points[nextJ].first) / 10.0f;
            float y1 = static_cast<float>(section1.points[nextJ].second) / 10.0f;
            float z1 = static_cast<float>(section1.Z) / 10.0f;
            float x2 = static_cast<float>(section2.points[j].first) / 10.0f;
            float y2 = static_cast<float>(section2.points[j].second) / 10.0f;
            float z2 = static_cast<float>(section2.Z) / 10.0f;
            float x3 = static_cast<float>(section2.points[nextJ].first) / 10.0f;
            float y3 = static_cast<float>(section2.points[nextJ].second) / 10.0f;
            float z3 = static_cast<float>(section2.Z) / 10.0f;
            // First triangle
            vertices.push_back(x0); vertices.push_back(y0); vertices.push_back(z0);
            vertices.push_back(x1); vertices.push_back(y1); vertices.push_back(z1);
            vertices.push_back(x2); vertices.push_back(y2); vertices.push_back(z2);
            // Second triangle
            vertices.push_back(x1); vertices.push_back(y1); vertices.push_back(z1);
            vertices.push_back(x3); vertices.push_back(y3); vertices.push_back(z3);
            vertices.push_back(x2); vertices.push_back(y2); vertices.push_back(z2);
        }
    }

    // Add end caps similarly
    // First end cap
    {
        const auto& section = cylinderSections.front();
        size_t M = section.points.size();
        float z0 = static_cast<float>(section.Z) / 10.0f;
        float centerX = 0.0f;
        float centerY = 0.0f;
        for (const auto& p : section.points) {
            centerX += static_cast<float>(p.first) / 10.0f;
            centerY += static_cast<float>(p.second) / 10.0f;
        }
        centerX /= M;
        centerY /= M;
        for (size_t j = 0; j < M; ++j) {
            size_t nextJ = (j + 1) % M;
            float x1 = static_cast<float>(section.points[j].first) / 10.0f;
            float y1 = static_cast<float>(section.points[j].second) / 10.0f;
            float x2 = static_cast<float>(section.points[nextJ].first) / 10.0f;
            float y2 = static_cast<float>(section.points[nextJ].second) / 10.0f;
            // Triangle
            vertices.push_back(centerX); vertices.push_back(centerY); vertices.push_back(z0);
            vertices.push_back(x1); vertices.push_back(y1); vertices.push_back(z0);
            vertices.push_back(x2); vertices.push_back(y2); vertices.push_back(z0);
        }
    }
    // Last end cap
    {
        const auto& section = cylinderSections.back();
        size_t M = section.points.size();
        float z0 = static_cast<float>(section.Z) / 10.0f;
        float centerX = 0.0f;
        float centerY = 0.0f;
        for (const auto& p : section.points) {
            centerX += static_cast<float>(p.first) / 10.0f;
            centerY += static_cast<float>(p.second) / 10.0f;
        }
        centerX /= M;
        centerY /= M;
        for (size_t j = 0; j < M; ++j) {
            size_t nextJ = (j + 1) % M;
            float x1 = static_cast<float>(section.points[j].first) / 10.0f;
            float y1 = static_cast<float>(section.points[j].second) / 10.0f;
            float x2 = static_cast<float>(section.points[nextJ].first) / 10.0f;
            float y2 = static_cast<float>(section.points[nextJ].second) / 10.0f;
            // Triangle
            vertices.push_back(centerX); vertices.push_back(centerY); vertices.push_back(z0);
            vertices.push_back(x2); vertices.push_back(y2); vertices.push_back(z0);
            vertices.push_back(x1); vertices.push_back(y1); vertices.push_back(z0);
        }
    }

    // Upload data to buffers
    glGenVertexArrays(1, &cylinderVAO);
    glGenBuffers(1, &cylinderVBO);
    glBindVertexArray(cylinderVAO);
    glBindBuffer(GL_ARRAY_BUFFER, cylinderVBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}




int main() {
    system("chcp 1251");
    logFile.open("log.txt");
    if (!logFile.is_open()) {
        std::cerr << "Не удалось открыть файл для логирования." << std::endl;
        return -1;
    }
    log(INFO, "Запуск программы.");
    std::ifstream file("model3.lprf", std::ios::binary);
    if (!file.is_open()) {
        log(ERROR, "Не удалось открыть файл.");
        return -1;
    }
    header = readHeader(file); // Инициализируем глобальную переменную
    log(INFO, "Файл версии: " + std::to_string(header.version));
    log(INFO, "Количество сечений: " + std::to_string(header.N));
    file.seekg(130, std::ios::beg);
    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(130, std::ios::beg);
    log(INFO, "Размер файла: " + std::to_string(fileSize) + " байт");
    sections = readSections(file, header.N, fileSize);
    file.close();
    if (!glfwInit()) {
        log(ERROR, "Failed to initialize GLFW");
        return -1;
    }
    GLFWwindow* window = glfwCreateWindow(800, 600, "3D Object Viewer", nullptr, nullptr);
    if (!window) {
        log(ERROR, "Failed to create GLFW window");
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    if (glewInit() != GLEW_OK) {
        log(ERROR, "Failed to initialize GLEW");
        return -1;
    }
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glUseProgram(shaderProgram);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    size_t BZ = createBuffersFromSections();
    glEnable(GL_DEPTH_TEST);
    while (!glfwWindowShouldClose(window)) {
        processInput(window);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Set wireframe mode
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(positionX, 0.0f, 0.0f));
        model = glm::rotate(model, rotationX, glm::vec3(1.0f, 0.0f, 0.0f));
        model = glm::rotate(model, rotationY, glm::vec3(0.0f, 1.0f, 0.0f));
        model = glm::scale(model, glm::vec3(scale, scale, scale));
        glm::mat4 view = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -500.0f));
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 1000.0f);
        GLuint modelLoc = glGetUniformLocation(shaderProgram, "model");
        GLuint viewLoc = glGetUniformLocation(shaderProgram, "view");
        GLuint projectionLoc = glGetUniformLocation(shaderProgram, "projection");
        GLuint colorLoc = glGetUniformLocation(shaderProgram, "ourColor");

        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));

        // Render original model
        glUniform4f(colorLoc, 1.0f, 1.0f, 1.0f, 1.0f); // White color
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(BZ / 3));
        glBindVertexArray(0);

        // Render cylinder model if created
        if (cylinderCreated) {
            glUniform4f(colorLoc, 1.0f, 0.0f, 0.0f, 1.0f); // Red color
            glBindVertexArray(cylinderVAO);
            // Исправлено: добавлен расчет количества вершин
            size_t numVertices = (cylinderSections.size() - 1) * numSegments * 6 + numSegments * 6 * 2;
            glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(numVertices));
            glBindVertexArray(0);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwTerminate();
    logFile.close();
    return 0;
}
